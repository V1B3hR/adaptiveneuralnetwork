"""Data loading and preprocessing utilities."""

from .kaggle_datasets import (
    create_text_classification_dataset,
    get_dataset_info,
    load_annomi_dataset,
    load_autvi_dataset,
    load_digakust_dataset,
    load_mental_health_faqs_dataset,
    load_social_media_sentiment_dataset,
    load_vr_driving_dataset,
    print_dataset_info,
)
from .optimized_datasets import (
    OptimizedDatasetWrapper,
    PreallocatedBuffer,
    VectorizedDataset,
    create_optimized_loader,
    optimize_dataset,
    vectorized_collate_fn,
)
from .streaming_datasets import (
    HuggingFaceDatasetWrapper,
    StreamingConfig,
    StreamingDatasetWrapper,
    UnifiedDatasetInterface,
    UnifiedDatasetManager,
    WebDatasetWrapper,
)

# Optional video streaming components
try:
    from .video_streaming import (
        FrameInfo,
        OpenCVVideoLoader,
        VideoConfig,
        VideoFrameProcessor,
        VideoStreamDataset,
        VideoStreamLoader,
        create_file_stream,
        create_rtsp_stream,
        create_video_stream,
        create_webcam_stream,
    )
    _video_streaming_available = True
except ImportError:
    _video_streaming_available = False

__all__ = [
    'load_annomi_dataset',
    'load_mental_health_faqs_dataset',
    'load_social_media_sentiment_dataset',
    'load_vr_driving_dataset',
    'load_autvi_dataset',
    'load_digakust_dataset',
    'create_text_classification_dataset',
    'print_dataset_info',
    'get_dataset_info',
    'StreamingConfig',
    'UnifiedDatasetInterface',
    'StreamingDatasetWrapper',
    'WebDatasetWrapper',
    'HuggingFaceDatasetWrapper',
    'UnifiedDatasetManager',
    'VectorizedDataset',
    'PreallocatedBuffer',
    'OptimizedDatasetWrapper',
    'vectorized_collate_fn',
    'create_optimized_loader',
    'optimize_dataset',
]

if _video_streaming_available:
    __all__.extend([
        'VideoConfig',
        'FrameInfo',
        'VideoStreamLoader',
        'OpenCVVideoLoader',
        'VideoFrameProcessor',
        'VideoStreamDataset',
        'create_video_stream',
        'create_webcam_stream',
        'create_file_stream',
        'create_rtsp_stream'
    ])
