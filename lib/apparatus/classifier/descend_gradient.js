const {
  Worker,
  isMainThread,
  parentPort,
  workerData,
} = require("node:worker_threads");

var sylvester = require("sylvester"),
  Matrix = sylvester.Matrix,
  Vector = sylvester.Vector;

function sigmoid(z) {
  return 1 / (1 + Math.exp(0 - z));
}

function hypothesis(theta, Observations) {
  return Observations.x(theta).map(sigmoid);
}

function cost(theta, Examples, classifications) {
  var hypothesisResult = hypothesis(theta, Examples);

  var ones = Vector.One(Examples.rows());
  var cost_1 = Vector.Zero(Examples.rows())
    .subtract(classifications)
    .elementMultiply(hypothesisResult.log());
  var cost_0 = ones
    .subtract(classifications)
    .elementMultiply(ones.subtract(hypothesisResult).log());

  return (1 / Examples.rows()) * cost_1.subtract(cost_0).sum();
}

function descendGradient(theta, Examples, classifications) {
  var maxIt = 500 * Examples.rows();
  var last;
  var current;
  var learningRate = 3;
  var learningRateFound = false;

  Examples = Matrix.One(Examples.rows(), 1).augment(Examples);
  theta = theta.augment([0]);

  while (!learningRateFound && learningRate !== 0) {
    var i = 0;
    last = null;

    while (true) {
      var hypothesisResult = hypothesis(theta, Examples);
      theta = theta.subtract(
        Examples.transpose()
          .x(hypothesisResult.subtract(classifications))
          .x(1 / Examples.rows())
          .x(learningRate)
      );
      current = cost(theta, Examples, classifications);

      i++;

      if (last) {
        if (current < last) learningRateFound = true;
        else break;

        if (last - current < 0.0001) break;
      }

      if (i >= maxIt) {
        throw "unable to find minimum";
      }

      last = current;
    }

    learningRate /= 3;
  }

  return theta.chomp(1);
}

if (isMainThread) {
  module.exports = {
    descendGradient,
    sigmoid,
  };
} else {
  // parentPort.on("message", (task) => {

  // });
  console.log("workerData", workerData);
  var zero = function () {
    return 0;
  };
  const thetas = [];
  const Examples = $M(workerData.examples);
  const Classifications = $M(workerData.allClassifications);
  for (let i = 0; i < workerData.classifications.length; i++) {
    const theta = Examples.row(1).map(zero);
    thetas.push(
      descendGradient(
        theta,
        Examples,
        Classifications.column(i + workerData.startIndex)
      )
    );
  }

  try {
    parentPort.postMessage(thetas);
  } catch (err) {
    console.error(err);
  }
  // const theta = task.Examples.row(1).map(zero);
  // parentPort.postMessage(
  //   descendGradient(theta, task.Examples, task.classifications)
  // );
}
