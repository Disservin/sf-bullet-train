use crate::outputs::OutputBuckets;
use bullet::{
    default::inputs::{Chess768, Factorised},
    nn::{
        optimiser::{AdamWParams, RangerOptimiser, RangerParams},
        Activation, ExecutionContext, Graph, InitSettings, NetworkBuilder, Node, Shape,
    },
    trainer::{
        default::{
            formats::sfbinpack::{
                chess::{piecetype::PieceType, r#move::MoveType},
                TrainingDataEntry,
            },
            inputs, loader, outputs, Trainer,
        },
        save::{Layout, QuantTarget, SavedFormat},
        schedule::{lr, wdl, TrainingSchedule, TrainingSteps},
        settings::LocalSettings,
    },
    NetworkTrainer,
};
use bulletformat::ChessBoard;
use halfkav2_hm::HalfKAv2_hm;

pub mod halfkav2_hm;

#[derive(Clone, Copy, Default)]
pub struct SfMaterialCount;
impl OutputBuckets<ChessBoard> for SfMaterialCount {
    const BUCKETS: usize = 8;

    fn bucket(&self, pos: &ChessBoard) -> u8 {
        let piece_count = pos.occ().count_ones() as u8 - 1;
        (piece_count / 4) as u8
    }
}

// type InputFeatures = Factorised<HalfKAv2_hm, Chess768>;
// type InputFeatures = inputs::ChessBucketsMergedKingsMirroredFactorised;
type InputFeatures = inputs::ChessBucketsMergedKingsMirrored;
const L1: usize = 3072;
const L2: usize = 15;
const L3: usize = 32;

const model_nnue2score: i32 = 600;
const model_weight_scale_hidden: i32 = 64;
const model_weight_scale_out: i32 = 16;
const model_quantized_one: i32 = 127;

const kWeightScaleHidden: i32 = model_weight_scale_hidden;
const kWeightScaleOut: i32 = model_nnue2score * model_weight_scale_out / model_quantized_one;
// const kWeightScale: i32 = kWeightScaleOut;
const kBiasScaleOut: i32 = model_weight_scale_out * model_nnue2score;
const kBiasScaleHidden: i32 = model_weight_scale_hidden * model_quantized_one;
// const kBiasScale: i32 = kBiasScaleOut;
const kMaxWeightOut: f32 = model_quantized_one as f32 / kWeightScaleOut as f32;
const kMaxWeightHidden: f32 = model_quantized_one as f32 / kWeightScaleHidden as f32;

fn main() {
    #[rustfmt::skip]
    let mut buckets = [
        0, 1, 2, 3,
        4, 5, 6, 7,
        8, 9, 10, 11,
        12, 13, 14, 15,
        16, 17, 18, 19,
        20, 21, 22, 23,
        24, 25, 26, 27,
        28, 29, 30, 31,
    ];

    let inputs = InputFeatures::new(buckets);
    // let inputs = Factorised::from_parts(HalfKAv2_hm::new(buckets), Chess768);

    let output_buckets = SfMaterialCount::default();
    let num_inputs = <InputFeatures as inputs::SparseInputType>::num_inputs(&inputs);
    let max_active = <InputFeatures as inputs::SparseInputType>::max_active(&inputs);
    let num_buckets = <SfMaterialCount as outputs::OutputBuckets<_>>::BUCKETS;

    let (mut graph, output_node) = build_network(num_inputs, max_active, num_buckets);

    initialize_weights(&mut graph, num_inputs);
    /*
    self.nnue2score = 600.0
    self.weight_scale_hidden = 64.0
    self.weight_scale_out = 16.0
    self.quantized_one = 127.0

    kWeightScaleHidden = model.weight_scale_hidden
    kWeightScaleOut = model.nnue2score * model.weight_scale_out / model.quantized_one
    kWeightScale = kWeightScaleOut if is_output else kWeightScaleHidden
    kBiasScaleOut = model.weight_scale_out * model.nnue2score
    kBiasScaleHidden = model.weight_scale_hidden * model.quantized_one
    kBiasScale = kBiasScaleOut if is_output else kBiasScaleHidden
    kMaxWeight = model.quantized_one / kWeightScale
     */

    let saved_format = vec![
        SavedFormat::new(
            "l0b",
            QuantTarget::I16(model_quantized_one as i16),
            Layout::Normal,
        ),
        SavedFormat::new(
            "l0w",
            QuantTarget::I16(model_quantized_one as i16),
            Layout::Normal,
        ),
        SavedFormat::new(
            "pst",
            QuantTarget::I32(model_nnue2score * kWeightScaleOut),
            Layout::Normal,
        ),
        SavedFormat::new("l1b", QuantTarget::I32(kBiasScaleHidden), Layout::Normal).add_transform(
            |graph, mut weights| {
                // let fact = graph.get_weights("l1_factb").get_dense_vals().unwrap();
                // add_factoriser(&mut weights, &fact, L2 + 1);
                weights
            },
        ),
        SavedFormat::new(
            "l1w",
            QuantTarget::I8(kWeightScaleHidden as i16),
            Layout::Transposed(Shape::new(NUM_BUCKETS * (L2 + 1), L1)),
        )
        .add_transform(|graph, mut weights| {
            // let fact = graph.get_weights("l1_factw").get_dense_vals().unwrap();
            // let fact = SavedFormat::transpose(Shape::new(L2 + 1, L1), &fact);
            // add_factoriser(&mut weights, &fact, (L2 + 1) * L1);

            for i in 0..weights.len() {
                weights[i] = weights[i].clamp(-kMaxWeightHidden, kMaxWeightHidden);
            }

            weights
        }),
        SavedFormat::new("l2b", QuantTarget::I32(kBiasScaleHidden), Layout::Normal),
        SavedFormat::new(
            "l2w",
            QuantTarget::I8(kWeightScaleHidden as i16),
            Layout::Transposed(Shape::new(NUM_BUCKETS * L3, L2 * 2)),
        )
        .add_transform(|graph, mut weights| {
            for i in 0..weights.len() {
                weights[i] = weights[i].clamp(-kMaxWeightHidden, kMaxWeightHidden);
            }

            weights
        }),
        SavedFormat::new("l3b", QuantTarget::I32(kBiasScaleOut), Layout::Normal),
        SavedFormat::new(
            "l3w",
            QuantTarget::I8(kWeightScaleOut as i16),
            Layout::Transposed(Shape::new(NUM_BUCKETS, L3)),
        )
        .add_transform(|graph, mut weights| {
            for i in 0..weights.len() {
                weights[i] = weights[i].clamp(-kMaxWeightOut, kMaxWeightOut);
            }

            weights
        }),
    ];

    let mut trainer = Trainer::<RangerOptimiser, _, _>::new(
        graph,
        output_node,
        RangerParams::default(),
        inputs,
        output_buckets,
        saved_format,
        false,
    );

    trainer.optimiser_mut().set_params_for_weight(
        "pst",
        RangerParams {
            decay: 0.0,
            ..Default::default()
        },
    );

    // trainer.mark_weights_as_input_factorised(&["l0w", "pst"]);

    println!("Params: {}", trainer.optimiser().graph.get_num_params());

    let schedule = TrainingSchedule {
        net_id: "test".to_string(),
        eval_scale: model_nnue2score as f32,
        steps: TrainingSteps {
            batch_size: 16_384,
            batches_per_superbatch: 1024,
            start_superbatch: 1,
            end_superbatch: 5,
        },
        wdl_scheduler: wdl::ConstantWDL { value: 0.0 },
        lr_scheduler: lr::StepLR {
            start: 0.001,
            gamma: 0.3,
            step: 60,
        },
        save_rate: 1,
    };

    let settings = LocalSettings {
        threads: 4,
        test_set: None,
        output_directory: "checkpoints/halfkav2_hm",
        batch_queue_size: 512,
    };

    let data_loader = {
        let file_path = "/mnt/g/stockfish-data/leela96-filt-v2.min.binpack";
        let buffer_size_mb = 1024;
        let threads = 8;
        fn filter(entry: &TrainingDataEntry) -> bool {
            entry.ply >= 16
                && !entry.pos.is_checked(entry.pos.side_to_move())
                && entry.score.unsigned_abs() <= 10000
                && entry.mv.mtype() == MoveType::Normal
                && entry.pos.piece_at(entry.mv.to).piece_type() == PieceType::None
        }

        loader::SfBinpackLoader::new(file_path, buffer_size_mb, threads, filter)
    };

    //trainer.profile_all_nodes();
    trainer.run(&schedule, &settings, &data_loader);
    //trainer.report_profiles();

    // trainer.load_from_checkpoint("./checkpoints/halfkav2_hm/test-1");

    // trainer.save_quantised("./checkpoints/halfkav2_hm/test-1/quantised.bin");

    let eval = model_nnue2score as f32
        * trainer.eval("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1 | 0 | 0.0");

    println!("Eval: {eval:.3}cp");

    let eval = model_nnue2score as f32
        * trainer.eval("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNB1KBNR w KQkq - 0 1 | 0 | 0.0");
    println!("Eval: {eval:.3}cp");

    let eval = model_nnue2score as f32
        * trainer.eval("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/4K3 w - - 0 1 | 0 | 0.0");
    println!("Eval: {eval:.3}cp");
}

fn build_network(num_inputs: usize, max_active: usize, num_buckets: usize) -> (Graph, Node) {
    let builder = NetworkBuilder::default();

    // inputs
    let stm = builder.new_sparse_input("stm", Shape::new(num_inputs, 1), max_active);
    let nstm = builder.new_sparse_input("nstm", Shape::new(num_inputs, 1), max_active);
    let targets = builder.new_dense_input("targets", Shape::new(1, 1));
    let buckets = builder.new_sparse_input("buckets", Shape::new(num_buckets, 1), 1);

    // trainable weights
    let l0 = builder.new_affine("l0", num_inputs, L1);
    let l1 = builder.new_affine("l1", L1, num_buckets * (L2 + 1));
    // let l1_fact = builder.new_affine("l1_fact", L1, L2 + 1);
    let l2 = builder.new_affine("l2", L2 * 2, num_buckets * L3);
    let l3 = builder.new_affine("l3", L3, num_buckets);
    let pst = builder.new_weights(
        "pst",
        Shape::new(num_buckets, num_inputs),
        InitSettings::Zeroed,
    );

    // inference
    let stm_subnet = l0.forward(stm).activate(Activation::CReLU);
    let ntm_subnet = l0.forward(nstm).activate(Activation::CReLU);
    let mut out = stm_subnet.concat(ntm_subnet);

    out = out.pairwise_mul_post_affine_dual();
    // out = l1.forward(out).select(buckets) + l1_fact.forward(out);
    out = l1.forward(out).select(buckets);

    let skip_neuron = out.slice_rows(15, 16);
    out = out.slice_rows(0, 15);

    out = out.concat(out.activate(Activation::Square));
    out = out.activate(Activation::CReLU);

    out = l2.forward(out).select(buckets).activate(Activation::SCReLU);
    out = l3.forward(out).select(buckets);

    let stm_pst = pst.matmul(stm).select(buckets);
    let ntm_pst = pst.matmul(nstm).select(buckets);
    let pst_out = stm_pst.linear_comb(0.5, ntm_pst, -0.5);
    out = out + skip_neuron + pst_out;

    let pred = out.activate(Activation::Sigmoid);
    pred.mpe(targets, 2.6);

    // graph, output node
    let output_node = out.node();
    (builder.build(ExecutionContext::default()), output_node)
}

const NUM_BUCKETS: usize = <SfMaterialCount as outputs::OutputBuckets<_>>::BUCKETS;

fn add_factoriser(weights: &mut [f32], factoriser: &[f32], size: usize) {
    for i in 0..NUM_BUCKETS * size {
        weights[i] += factoriser[i % size]
    }
}

const NUM_SQ: usize = 64;
const NUM_PT_REAL: usize = 11;
const NUM_PT_VIRTUAL: usize = 12;
const NUM_PLANES_REAL: usize = NUM_SQ * NUM_PT_REAL;
const NUM_PLANES_VIRTUAL: usize = NUM_SQ * NUM_PT_VIRTUAL;
const NUM_INPUTS: usize = NUM_PLANES_REAL * NUM_SQ / 2;

const PAWN: i32 = 0;
const KNIGHT: i32 = 1;
const BISHOP: i32 = 2;
const ROOK: i32 = 3;
const QUEEN: i32 = 4;

fn halfka_psqts() -> Vec<f32> {
    let piece_values = [
        (PAWN, 126.0),
        (KNIGHT, 781.0),
        (BISHOP, 825.0),
        (ROOK, 1276.0),
        (QUEEN, 2538.0),
    ];

    let mut values = vec![0 as f32; NUM_INPUTS];

    for ksq in 0..64 {
        for s in 0..64 {
            for &(pt, val) in &piece_values {
                let idxw = HalfKAv2_hm::feature_index(true, ksq as u8, s as u8, pt as u8);
                let idxb = HalfKAv2_hm::feature_index(true, ksq as u8, s as u8, pt as u8 + 8);
                values[idxw] = -val;
                values[idxb] = val;
            }
        }
    }

    values
}
fn initialize_weights(graph: &mut Graph, num_inputs: usize) {
    let mut original = halfka_psqts();

    // let virtual_values = vec![0.0; NUM_PLANES_VIRTUAL];
    // original.extend_from_slice(&virtual_values);

    let mut values = original.repeat(8);

    for i in 0..values.len() {
        values[i] *= 1.0 / model_nnue2score as f32;
    }

    let weights = graph.get_weights("pst").values.batch_size();

    graph
        .get_weights_mut("pst")
        .load_from_slice(weights, &values)
        .unwrap();
}
