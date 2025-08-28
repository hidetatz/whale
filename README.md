whale DNN compiler.

## devel

Initialize the environment:

```sh
python3.13 -m venv .env
source .env/bin/activate
pip install mypy black isort torch # autoformat, lint, test
```

run autoformat:

```sh
./task fmt
```

run type check:

```sh
./task check
```

run unittest:

```sh
./task test
```
