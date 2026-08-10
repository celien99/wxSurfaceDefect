PREPROCESSING_SCHEMA_VERSION = 3
PREPROCESSING_REGISTRY_SCHEMA_VERSION = 1
PREPROCESSING_DIRECTORY = "preprocessing"
PREPROCESSING_REGISTRY_FILE = "preprocessing_registry.json"
PREPROCESSING_CONFIG_FILE = "preprocessing.yaml"
PREPROCESSING_MANIFEST_FILE = "preprocessing_manifest.json"
PROTOTYPES_FILE = "foreground_prototypes.pt"
REFERENCE_TEMPLATE_FILE = "reference_feature_template.pt"
REFERENCE_MASK_FILE = "reference_foreground.rle"

CONFIG_KEYS = (
    "schema_version",
    "array_color_space",
    "input_scale",
    "mean",
    "std",
    "reference_manifest",
    "dino_backbone_name",
    "dino_feature_layer",
    "working_longest_edge",
    "boundary_expand_ratio",
    "min_dino_matches",
    "min_dino_inlier_ratio",
    "max_dino_reprojection_ratio",
    "max_area_ratio_deviation",
    "min_reference_coverage",
)
