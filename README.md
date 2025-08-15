whale DNN compiler.

## devel

Initialize the environment:

```sh
python3.13 -m venv .env
source .env/bin/activate
pip install black isort torch # autoformat, test
```

run autoformat:

```sh
./task fmt
```

run unittest:

```sh
./task test
```
