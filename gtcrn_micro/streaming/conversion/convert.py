# Original author Xiaobin Rong
# Source: SEtrain: https://github.com/Xiaobin-Rong/gtcrn

import torch


def convert_to_stream(stream_model, model) -> None:
    """Copy model weights from offline model to streaming model.

    Args:
        stream_model : New streaming model to copy weights to
        model : Original offline model to copy weights from

    Raises:
        ValueError: Raise value error if key in state dicts don't match at all
    """
    og_state_dict = model.state_dict()
    new_state_dict = stream_model.state_dict()

    for key in stream_model.state_dict().keys():
        # if params match, copy over without issue
        if key in og_state_dict.keys():
            new_state_dict[key] = og_state_dict[key]

        # handling if model wraps extra conv#d in state_dict
        elif key.replace("Conv1d.", "") in og_state_dict.keys():
            new_state_dict[key] = og_state_dict[key.replace("Conv1d.", "")]

        elif key.replace("Conv2d.", "") in og_state_dict.keys():
            new_state_dict[key] = og_state_dict[key.replace("Conv2d.", "")]

        elif key.replace(".deconv", "") in og_state_dict.keys():
            new_state_dict[key] = og_state_dict[key.replace(".deconv", "")]

        # adjusting the weight layouts for streaming ConvTranspose2d
        elif key.replace("ConvTranspose2d.", "") in og_state_dict.keys():
            if key.endswith("weight"):
                if (
                    new_state_dict[key].shape
                    != og_state_dict[key.replace("ConvTranspose2d.", "")].shape
                ):
                    new_state_dict[key] = torch.flip(
                        og_state_dict[key.replace("ConvTranspose2d.", "")].permute(
                            [1, 0, 2, 3]
                        ),
                        dims=[-2, -1],
                    )
                else:
                    new_state_dict[key] = torch.flip(
                        og_state_dict[key.replace("ConvTranspose2d.", "")],
                        dims=[-2, -1],
                    )
            else:
                new_state_dict[key] = og_state_dict[key.replace("ConvTranspose2d.", "")]

        else:
            raise (ValueError("Key error!"))

    stream_model.load_state_dict(new_state_dict)
