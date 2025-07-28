from functools import lru_cache
from typing import List, Tuple

import numpy
from tqdm import tqdm

from facefusion import inference_manager, state_manager, wording
from facefusion.download import conditional_download_hashes, conditional_download_sources, resolve_download_url
from facefusion.execution import has_execution_provider
from facefusion.filesystem import resolve_relative_path
from facefusion.thread_helper import conditional_thread_semaphore
from facefusion.types import Detection, DownloadScope, DownloadSet, ExecutionProvider, Fps, InferencePool, ModelSet, VisionFrame
from facefusion.vision import detect_video_fps, fit_frame, read_image, read_video_frame

STREAM_COUNTER = 0


@lru_cache(maxsize = None)
def create_static_model_set(download_scope : DownloadScope) -> ModelSet:
	return {}


def get_inference_pool() -> InferencePool:
	return None


def clear_inference_pool() -> None:
	pass


def resolve_execution_providers() -> List[ExecutionProvider]:
	if has_execution_provider('coreml'):
		return [ 'cpu' ]
	return state_manager.get_item('execution_providers')


def collect_model_downloads() -> Tuple[DownloadSet, DownloadSet]:
	return {}, {}


def pre_check() -> bool:
	model_hash_set, model_source_set = collect_model_downloads()

	return conditional_download_hashes(model_hash_set) and conditional_download_sources(model_source_set)


def analyse_stream(vision_frame : VisionFrame, video_fps : Fps) -> bool:
	global STREAM_COUNTER

	STREAM_COUNTER = STREAM_COUNTER + 1
	if STREAM_COUNTER % int(video_fps) == 0:
		return analyse_frame(vision_frame)
	return False


def analyse_frame(vision_frame : VisionFrame) -> bool:
	return detect_nsfw(vision_frame)


@lru_cache(maxsize = None)
def analyse_image(image_path : str) -> bool:
	vision_frame = read_image(image_path)
	return analyse_frame(vision_frame)


@lru_cache(maxsize = None)
def analyse_video(video_path : str, trim_frame_start : int, trim_frame_end : int) -> bool:
	video_fps = detect_video_fps(video_path)
	frame_range = range(trim_frame_start, trim_frame_end)
	rate = 0.0
	total = 0
	counter = 0

	with tqdm(total = len(frame_range), desc = wording.get('analysing'), unit = 'frame', ascii = ' =', disable = state_manager.get_item('log_level') in [ 'warn', 'error' ]) as progress:

		for frame_number in frame_range:
			if frame_number % int(video_fps) == 0:
				vision_frame = read_video_frame(video_path, frame_number)
				total += 1
				if analyse_frame(vision_frame):
					counter += 1
			if counter > 0 and total > 0:
				rate = counter / total * 100
			progress.set_postfix(rate = rate)
			progress.update()

	return bool(rate > 10.0)


def detect_nsfw(vision_frame : VisionFrame) -> bool:
	return False


def detect_with_nsfw_1(vision_frame : VisionFrame) -> bool:
	return False


def detect_with_nsfw_2(vision_frame : VisionFrame) -> bool:
	return False


def detect_with_nsfw_3(vision_frame : VisionFrame) -> bool:
	return False


def forward_nsfw(vision_frame : VisionFrame, nsfw_model : str) -> Detection:
	return [0.0, 0.0, 0.0, 0.0]


def prepare_detect_frame(temp_vision_frame : VisionFrame, model_name : str) -> VisionFrame:
	model_set = create_static_model_set('full').get(model_name)
	model_size = model_set.get('size')
	model_mean = model_set.get('mean')
	model_standard_deviation = model_set.get('standard_deviation')

	detect_vision_frame = fit_frame(temp_vision_frame, model_size)
	detect_vision_frame = detect_vision_frame[:, :, ::-1] / 255.0
	detect_vision_frame -= model_mean
	detect_vision_frame /= model_standard_deviation
	detect_vision_frame = numpy.expand_dims(detect_vision_frame.transpose(2, 0, 1), axis = 0).astype(numpy.float32)
	return detect_vision_frame
