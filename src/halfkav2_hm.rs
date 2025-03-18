use bullet::default::inputs::{Chess768, Factorised, Factorises, SparseInputType};
use bulletformat::ChessBoard;

#[derive(Clone, Copy, Debug)]
pub struct HalfKAv2_hm {}

impl HalfKAv2_hm {
    pub fn new(buckets: [usize; 32]) -> Self {
        Self {}
    }

    // Constants matching the C++ implementation
    const NUM_SQ: usize = 64;
    const NUM_PT: usize = 11;
    const NUM_PLANES: usize = Self::NUM_SQ * Self::NUM_PT;
    const INPUTS: usize = Self::NUM_PLANES * Self::NUM_SQ / 2;

    const MAX_ACTIVE_FEATURES: usize = 32;

    // King buckets array from C++
    #[rustfmt::skip]
    const KING_BUCKETS: [isize; 64] = [
        -1, -1, -1, -1, 31, 30, 29, 28,
        -1, -1, -1, -1, 27, 26, 25, 24,
        -1, -1, -1, -1, 23, 22, 21, 20,
        -1, -1, -1, -1, 19, 18, 17, 16,
        -1, -1, -1, -1, 15, 14, 13, 12,
        -1, -1, -1, -1, 11, 10, 9, 8,
        -1, -1, -1, -1, 7, 6, 5, 4,
        -1, -1, -1, -1, 3, 2, 1, 0
    ];

    // Direct translation of orient_flip_2 from C++
    fn orient_flip_2(color: bool, sq: u8, ksq: u8) -> u8 {
        let h = ksq % 8 < 4;
        let mut result = sq;

        if color {
            // Vertical flip for black
            result ^= 56;
        }

        if h {
            // Horizontal flip if king is on left half
            result ^= 7;
        }

        result
    }

    // Direct translation of feature_index from C++
    pub fn feature_index(color: bool, ksq: u8, sq: u8, piece: u8) -> usize {
        let o_ksq = Self::orient_flip_2(color, ksq, ksq);

        // Calculate piece index
        let piece_type = piece & 7; // Type only
        let piece_color = (piece & 8) > 0; // Color bit

        let mut p_idx = (piece_type * 2 + if piece_color != color { 1 } else { 0 }) as usize;

        // Pack the opposite king
        if p_idx == 13 {
            p_idx = 12;
        }

        // Get king bucket
        let king_bucket = Self::KING_BUCKETS[o_ksq as usize];
        assert!(king_bucket >= 0, "Invalid king position in buckets");

        // Orient the square
        let oriented_sq = Self::orient_flip_2(color, sq, ksq) as usize;

        // Calculate final feature index
        oriented_sq + (p_idx * Self::NUM_SQ) + (king_bucket as usize * Self::NUM_PLANES)
    }
}

impl SparseInputType for HalfKAv2_hm {
    type RequiredDataType = ChessBoard;

    fn num_inputs(&self) -> usize {
        HalfKAv2_hm::INPUTS
    }

    fn max_active(&self) -> usize {
        HalfKAv2_hm::MAX_ACTIVE_FEATURES
    }

    fn map_features<F: FnMut(usize, usize)>(&self, pos: &Self::RequiredDataType, mut f: F) {
        for (piece, square) in pos.into_iter() {
            let c = usize::from(piece & 8 > 0) != 0;

            // Calculate feature from our perspective
            let our_feature = HalfKAv2_hm::feature_index(c, pos.our_ksq(), square, piece);

            // Calculate feature from opponent's perspective
            let opp_feature = HalfKAv2_hm::feature_index(!c, pos.opp_ksq(), square, piece);

            f(our_feature, opp_feature);
        }
    }

    fn shorthand(&self) -> String {
        format!("{}hm", HalfKAv2_hm::INPUTS)
    }

    fn description(&self) -> String {
        "HalfKAv2 with horizontal mirror".to_string()
    }
}

impl Factorises<HalfKAv2_hm> for Chess768 {
    fn derive_feature(&self, _inputs: &HalfKAv2_hm, feat: usize) -> Option<usize> {
        // Extract piece index and square from the feature
        let feature_in_plane = feat % HalfKAv2_hm::NUM_PLANES;
        let p_idx = feature_in_plane / HalfKAv2_hm::NUM_SQ;
        let sq = feature_in_plane % HalfKAv2_hm::NUM_SQ;

        // Return the base feature
        Some(p_idx * HalfKAv2_hm::NUM_SQ + sq)
    }
}
