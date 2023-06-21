# Copyright (c) OpenMMLab. All rights reserved.
import os
import pathlib

import click
import mmcv
from mmcv.transforms import Compose
from mmdet.apis import inference_detector, init_detector
from mmdet.registry import VISUALIZERS
from mmengine.utils import track_iter_progress
from moviepy.editor import ImageSequenceClip


@click.command()
@click.argument(
    "input",
    nargs=1,
    required=True,
    type=click.Path(
        file_okay=False,
        dir_okay=True,
        exists=True,
        readable=True,
        path_type=pathlib.Path,
    ),
)
@click.argument(
    "output",
    nargs=1,
    required=True,
    type=click.Path(
        file_okay=False,
        dir_okay=True,
        path_type=pathlib.Path,
    ),
)
@click.argument(
    "config",
    nargs=1,
    required=True,
    type=click.Path(
        file_okay=True,
        dir_okay=False,
        exists=True,
        readable=True,
        path_type=pathlib.Path,
    ),
)
@click.argument(
    "checkpoint",
    nargs=1,
    required=True,
    type=click.Path(
        file_okay=True,
        dir_okay=False,
        exists=True,
        readable=True,
        path_type=pathlib.Path,
    ),
)
@click.option(
    "--extension",
    "-x",
    type=str,
    nargs=1,
    default="mp4",
    help="The filename extension filter.",
)
@click.option(
    "--score-thr",
    type=float,
    nargs=1,
    default=0.3,
    help="Bbox score threshold.",
)
def main(input, output, config, checkpoint, extension, score_thr):
    model = init_detector(str(config), str(checkpoint), device="cuda")
    model.cfg.test_dataloader.dataset.pipeline[0].type = "LoadImageFromNDArray"

    test_pipeline = Compose(model.cfg.test_dataloader.dataset.pipeline)

    visualizer = VISUALIZERS.build(model.cfg.visualizer)
    visualizer.dataset_meta = model.dataset_meta

    labels = model.dataset_meta["classes"]

    for action in input.iterdir():
        if not os.path.isdir(action):
            continue

        for video in action.iterdir():
            if video.suffix != f".{extension}":
                continue

            video_reader = mmcv.VideoReader(str(video))
            output_frames = []
            output_video_path = output / action.name / video.with_suffix(".mp4").name
            first = True

            output_video_path.parent.mkdir(parents=True, exist_ok=True)

            for frame in track_iter_progress(video_reader):
                result = inference_detector(model, frame, test_pipeline=test_pipeline)

                # result.pred_instances = [
                #     i for i in result.pred_instances if i.labels[0] == 0
                # ]

                # for item in result.pred_instances:
                # if item.labels[0] != 0:
                # del item

                # for item in result.pred_instances:
                #     if item.labels[0] != 0:
                #         result.pred_instances = item
                # print(labels[item.labels[0]])

                pred_copy = result.pred_instances

                for item in pred_copy:
                    if item.labels[0] != 0:
                        del item

                result.pred_instances = pred_copy

                visualizer.add_datasample(
                    name="video",
                    image=frame,
                    data_sample=result,
                    draw_gt=False,
                    show=False,
                    pred_score_thr=score_thr,
                )

                frame = visualizer.get_image()

                output_frames.append(frame)

                # if first:
                #     print(type(result))
                #     print(type(result.pred_instances))
                #     first = False

            ImageSequenceClip(
                output_frames, fps=video_reader.fps
            ).without_audio().write_videofile(str(output_video_path))

            break
        break


if __name__ == "__main__":
    main()
