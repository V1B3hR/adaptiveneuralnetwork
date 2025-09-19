"""Data loading and preprocessing utilities."""

from .kaggle_datasets import (
    load_annomi_dataset,
    load_mental_health_faqs_dataset,
    load_social_media_sentiment_dataset,
    create_text_classification_dataset,
    print_dataset_info,
    get_dataset_info
)

from .streaming_datasets import (
    StreamingConfig,
    UnifiedDatasetInterface,
    StreamingDatasetWrapper,
    WebDatasetWrapper,
    HuggingFaceDatasetWrapper,
    UnifiedDatasetManager
)

# Optional video streaming components
try:
    from .video_streaming import (
        VideoConfig,
        FrameInfo,
        VideoStreamLoader,
        OpenCVVideoLoader,
        VideoFrameProcessor,
        VideoStreamDataset,
        create_video_stream,
        create_webcam_stream,
        create_file_stream,
        create_rtsp_stream
    )
    _video_streaming_available = True
except ImportError:
    _video_streaming_available = False

__all__ = [
    'load_annomi_dataset',
    'load_mental_health_faqs_dataset',
    'load_social_media_sentiment_dataset',
    'create_text_classification_dataset',
    'print_dataset_info',
    'get_dataset_info',
    'StreamingConfig',
    'UnifiedDatasetInterface',
    'StreamingDatasetWrapper',
    'WebDatasetWrapper',
    'HuggingFaceDatasetWrapper',
    'UnifiedDatasetManager'
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