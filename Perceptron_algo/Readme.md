## Intuition

The Perceptron algorithm models a linear decision boundary, making it suitable for simple classification tasks. It serves as a foundation for understanding more complex neural network architectures.


lossfunction.py:

Forward Propagation:
 the line z = w1*X[i][0] + w2*X[i][1] + b performs forward propagation. It computes the linear weighted sum z

Weight Updates:

The if z*y[i] < 0 condition identifies misclassifications and updates weights/biases using the perceptron learning rule.

Visualization:
At the end, the decision boundary is plotted based on learned weights and bias.



perceptron.py:
Forward Propagation:
Uses scikit-learn's Perceptron, which internally implements forward propagation to compute the weighted sum and classify data.

Weight Updates:
Scikit-learn's implementation handles this automatically during the .fit() method.

Decision Boundary:
The learned weights and bias are visualized using decision regions.
