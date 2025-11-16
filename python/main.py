import onnx


def main():
    print("python!")
    print("ONNX info:")
    gtcrn_onnx = onnx.load("./models/onnx/gtcrn.onnx")

    try:
        onnx.checker.check_model
        print("valid")
    except Exception as e:
        print(f"Model check failed with: {e}")

    # iterate through ops
    ops = set()
    for node in gtcrn_onnx.graph.node:
        ops.add(node.op_type)
        # print(
        #     "name=%r type=%r input=%r output=%r"
        #     % (node.name, node.op_type, node.input, node.output)
        # )
    for op in ops:
        print("type=%r" % (op))


if __name__ == "__main__":
    main()
