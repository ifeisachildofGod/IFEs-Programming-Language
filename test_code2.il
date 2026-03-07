func say(var data: string) => null {}

func main(const args: list[string]) {
    var x: int = 2;
    var y: float = 1.1;

    x += 1;

    var name: string = "Ife is {x}";
    name.replace("f", "d");

    var data: list[string] = ["hello", "he"];

    forevery (var value: string) in data {
        say("{value} is a value that you cant do shit about it");
    }

    while (1) {
        break;
    }

    for (var i: int = 0; i += 1; i < 1) {
        say(i);
    }

    if (name == "ife" and x == 2) {
        say("{name} is ife and {x} is 2");
    }
}
