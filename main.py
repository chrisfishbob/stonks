def main() -> None:
    result = add_two_numbers("a", "b")
    print(result)


def add_two_numbers(a: int, b: int) -> int:
    return a + b


if __name__ == "__main__":
    main()
