import os


def parse_file(lines, overwrite=["1", "-1", "-1", "-1\n"]):
    output = []
    for line in lines:
        parsed = line.split(",")
        parsed[-3:] = overwrite
        new_line = ",".join(parsed)
        output.append(new_line)

    return output


def main():
    results_dir = "results"
    files = os.listdir(results_dir)

    for f in files:
        print(f"Parsing {f}")
        f = os.path.join(results_dir, f)
        with open(f, "r") as of:
            data = of.readlines()

        output = parse_file(data)

        print(f"Overwriting {f}")
        with open(f, "w") as of:
            of.writelines(output)


if __name__ == "__main__":
    main()
