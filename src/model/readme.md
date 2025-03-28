### 生成命令

```
../../3rd/grpc/bin/protoc -I . --grpc_out=. --plugin=protoc-gen-grpc=../../3rd/grpc/bin/grpc_cpp_plugin task_exchange.proto
```

```
../../3rd/grpc/bin/protoc -I . --cpp_out=. task_exchange.proto
```