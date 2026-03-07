func _raw(const code: string = "9") => null {}

func say(var data: string = "9") => null {
    _raw("print(data)");
}

func main(const args: list[string]) {
    say("whatever");
}