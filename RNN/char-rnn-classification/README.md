# 使用 RNN 网络对姓名进行分类

## v0

1. 基于此[文章](https://toutiao.io/posts/sp0at6/preview)
2. 测试集准确率：0.55
3. `jupyer`中训练有问题，暂未找到原因
4. 使用`CrossEntropyLoss`损失函数，不需要在网络最后一层添加`softmax`层，
   `CrossEntropyLoss`已经内置了`softmax`的运算操作
5. 