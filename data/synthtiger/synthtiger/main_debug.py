"""
SynthTIGER
Copyright (c) 2021-present NAVER Corp.
MIT license
"""

import argparse
import pprint
import time

import synthtiger


def run(args):
    if args.config is not None:
        config = synthtiger.read_config(args.config)

    pprint.pprint(config)

    synthtiger.set_global_random_seed(args.seed)
    template = synthtiger.read_template(args.script, args.name, config)
    generator = synthtiger.generator(
        args.script,
        args.name,
        config=config,
        count=args.count,
        worker=args.worker,
        seed=args.seed,
        retry=True,
        verbose=args.verbose,
    )

    if args.output is not None:
        template.init_save(args.output)

    for idx, (task_idx, data) in enumerate(generator):
        if args.output is not None:
            template.save(args.output, data, task_idx)
        print(f"Generated {idx + 1} data (task {task_idx})")

    if args.output is not None:
        template.end_save(args.output)


def parse_args():
    parser = argparse.ArgumentParser()
    # 输出目录：默认results_zh_mtl（匹配目标启动参数-o）
    parser.add_argument(
        "-o",
        "--output",
        metavar="DIR",
        type=str,
        default="results_zh_mtl",
        help="Directory path to save data. [default: results_zh_mtl]",
    )
    # 生成数量：保持默认100，可通过命令行覆盖
    parser.add_argument(
        "-c",
        "--count",
        metavar="NUM",
        type=int,
        default=100,
        help="Number of output data. [default: 100]",
    )
    # 工作进程数：默认4（匹配目标启动参数-w）
    parser.add_argument(
        "-w",
        "--worker",
        metavar="NUM",
        type=int,
        default=1,
        help="Number of workers. If 0, It generates data in the main process. [default: 4]",
    )
    # 随机种子：默认None（保持原逻辑）
    parser.add_argument(
        "-s",
        "--seed",
        metavar="NUM",
        type=int,
        default=None,
        help="Random seed. [default: None]",
    )
    #  verbose模式：默认开启（匹配目标启动参数-v）
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        default=True,
        help="Print error messages while generating data. [default: True]",
    )
    # 脚本路径：默认examples/synthtiger/template_zn_multiline.py（位置参数1）
    parser.add_argument(
        "script",
        metavar="SCRIPT",
        type=str,
        nargs="?",  # 设为可选参数，允许无传参时使用默认值
        default="examples/synthtiger/template_zn_multiline.py",
        help="Script file path. [default: examples/synthtiger/template_zn_multiline.py]",
    )
    # 模板类名：默认SynthTigerZH（位置参数2）
    parser.add_argument(
        "name",
        metavar="NAME",
        type=str,
        nargs="?",  # 设为可选参数
        default="SynthTigerZH",
        help="Template class name. [default: SynthTigerZH]",
    )
    # 配置文件路径：默认examples/synthtiger/config_multiline_zh.yaml（位置参数3）
    parser.add_argument(
        "config",
        metavar="CONFIG",
        type=str,
        nargs="?",
        default="examples/synthtiger/config_multiline_zh_debug.yaml",
        help="Config file path. [default: examples/synthtiger/config_multiline_zh.yaml]",
    )
    args = parser.parse_args()

    pprint.pprint(vars(args))

    return args


def main():
    start_time = time.time()
    args = parse_args()
    run(args)
    end_time = time.time()
    print(f"{end_time - start_time:.2f} seconds elapsed")


if __name__ == "__main__":
    main()
    