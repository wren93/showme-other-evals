import collections
import io
import json
import logging
import os

import fire

from capture_metric.capture import CAPTURE
from PIL import Image
from pycocoevalcap.eval import Bleu, Cider, COCOEvalCap, Meteor, Rouge
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from pycocotools.coco import COCO

eval_logger = logging.getLogger("detailcaps")


# def generate_submission_file(file_name, args, subpath="submissions"):
#     if args.output_path is None:
#         # If no output path is specified, use current directory
#         path = subpath
#     else:
#         path = os.path.join(args.output_path, subpath)
#     os.makedirs(path, exist_ok=True)
#     path = os.path.join(path, file_name)
#     return os.path.abspath(path)


def detailcaps_aggregation_result(results, metric, args=None):
    scorers = [
        (Bleu(4), "Bleu_1"),
        (Bleu(4), "Bleu_2"),
        (Bleu(4), "Bleu_3"),
        (Bleu(4), "Bleu_4"),
        (Meteor(), "METEOR"),
        (Rouge(), "ROUGE_L"),
        (Cider(), "CIDEr"),
        (CAPTURE(), "CAPTURE"),
    ]
    scorers_dict = {s[1]: s for s in scorers}

    stored_results = []
    # In order to make the coco eval tools to successfully create index
    # We need at least two dict in the dataset
    # 'annotation' and 'images'
    # 'annotation' exactly reproduce the original annotation
    # 'images' however only need the image id which is contained in the file name
    dataset = {"annotations": [], "images": []}
    idx = 0

    for result in results:
        stored_results.append(
            {"image_id": result["image_id"], "caption": result["pred"]}
        )
        for a in result["answer"]:
            dataset["annotations"].append(
                {"image_id": result["image_id"], "caption": a, "id": idx}
            )
            idx += 1
        dataset["images"].append({"id": result["image_id"]})

    coco = COCO()
    # Manually create index here
    coco.dataset = dataset
    coco.createIndex()

    detailcaps_result = coco.loadRes(stored_results)
    detailcaps_eval = COCOEvalCap(coco, detailcaps_result)

    imgIds = detailcaps_eval.params["image_id"]
    gts = {}
    res = {}
    for imgId in imgIds:
        gts[imgId] = detailcaps_eval.coco.imgToAnns[imgId]
        res[imgId] = detailcaps_eval.cocoRes.imgToAnns[imgId]

    eval_logger.info("tokenization...")
    tokenizer = PTBTokenizer()

    if metric == "CAPTURE":
        reorg_gts, reorg_res = collections.defaultdict(list), collections.defaultdict(
            list
        )
        for _, samples in gts.items():
            for sample in samples:
                reorg_gts[sample["image_id"]].append(sample["caption"])
        for _, samples in res.items():
            for sample in samples:
                reorg_res[sample["image_id"]].append(sample["caption"])
        gts, res = reorg_gts, reorg_res
    else:
        gts = tokenizer.tokenize(gts)
        res = tokenizer.tokenize(res)

    eval_logger.info(f"Computing {metric} scores...")

    # if int(os.environ.get("RANK", 0)) == 0:
    #     from IPython import embed; embed()
    # else:
    #     import time; time.sleep(1200)

    score, scores = scorers_dict[metric][0].compute_score(gts, res)
    # When metric is one of the Bleu, score will be a list
    if type(score) == list:
        n = int(metric.split("_")[-1])
        score = score[n - 1]

    print(f"{metric} score: {score}")
    # path = generate_submission_file(f"detailcaps_val_{metric}_scores.json", args)
    # eval_logger.info("Storing prediction that can be submitted to the server ...")
    # with open(path, "w") as f:
    #     json.dump(stored_results, f, indent=4)
    # eval_logger.info(f"Your result has been saved to {path}.")

    return score


def detailcaps_bleu4(results, args=None):
    return detailcaps_aggregation_result(results, "Bleu_4", args)


def detailcaps_bleu3(results, args=None):
    return detailcaps_aggregation_result(results, "Bleu_3", args)


def detailcaps_bleu2(results, args=None):
    return detailcaps_aggregation_result(results, "Bleu_2", args)


def detailcaps_bleu1(results, args=None):
    return detailcaps_aggregation_result(results, "Bleu_1", args)


def detailcaps_meteor(results, args=None):
    return detailcaps_aggregation_result(results, "METEOR", args)


def detailcaps_rougel(results, args=None):
    return detailcaps_aggregation_result(results, "ROUGE_L", args)


def detailcaps_cider(results, args=None):
    return detailcaps_aggregation_result(results, "CIDEr", args)


def detailcaps_spice(results, args=None):
    return detailcaps_aggregation_result(results, "SPICE", args)


def detailcaps_capture(results, args=None):
    return detailcaps_aggregation_result(results, "CAPTURE", args)


def main(jsonl_path):
    with open(jsonl_path, "r") as f:
        data = [json.loads(line) for line in f]

    results = []
    for i, doc in enumerate(data):
        new_item = {}
        new_item["answer"] = [
            doc["GT_Caption_GPT4O".lower()],
            doc["GT_Caption_GPT4V".lower()],
            doc["GT_Caption_Gemini15Pro".lower()],
        ]
        new_item["pred"] = doc["generated_text"]
        new_item["image_id"] = i
        results.append(new_item)

    bleu1 = detailcaps_bleu1(results)
    bleu2 = detailcaps_bleu2(results)
    bleu3 = detailcaps_bleu3(results)
    bleu4 = detailcaps_bleu4(results)
    rougel = detailcaps_rougel(results)
    meteor = detailcaps_meteor(results)
    cider = detailcaps_cider(results)
    capture = detailcaps_capture(results)
    print(
        "=============================== Evaluation results: ==============================="
    )
    print(f"BLEU-1: {bleu1}")
    print(f"BLEU-2: {bleu2}")
    print(f"BLEU-3: {bleu3}")
    print(f"BLEU-4: {bleu4}")
    print(f"ROUGE-L: {rougel}")
    print(f"METEOR: {meteor}")
    print(f"CIDEr: {cider}")
    print(f"CAPTURE: {capture}")


if __name__ == "__main__":
    fire.Fire(main)
