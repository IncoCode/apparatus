/*
Copyright (c) 2011, Chris Umbel

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

var util = require("util"),
  Classifier = require("./classifier");
const { Worker } = require("node:worker_threads");
const { sigmoid } = require("./descend_gradient");
const path = require("path");

var sylvester = require("sylvester"),
  Matrix = sylvester.Matrix,
  Vector = sylvester.Vector;

// function sigmoid(z) {
//   return 1 / (1 + Math.exp(0 - z));
// }

// function hypothesis(theta, Observations) {
//     return Observations.x(theta).map(sigmoid);
// }

// function cost(theta, Examples, classifications) {
//     var hypothesisResult = hypothesis(theta, Examples);

//     var ones = Vector.One(Examples.rows());
//     var cost_1 = Vector.Zero(Examples.rows()).subtract(classifications).elementMultiply(hypothesisResult.log());
//     var cost_0 = ones.subtract(classifications).elementMultiply(ones.subtract(hypothesisResult).log());

//     return (1 / Examples.rows()) * cost_1.subtract(cost_0).sum();
// }

// function descendGradient(theta, Examples, classifications) {
//     var maxIt = 500 * Examples.rows();
//     var last;
//     var current;
//     var learningRate = 3;
//     var learningRateFound = false;

//     Examples = Matrix.One(Examples.rows(), 1).augment(Examples);
//     theta = theta.augment([0]);

//     while(!learningRateFound && learningRate !== 0) {
//         var i = 0;
//         last = null;

//         while(true) {
//             var hypothesisResult = hypothesis(theta, Examples);
//             theta = theta.subtract(Examples.transpose().x(
//             hypothesisResult.subtract(classifications)).x(1 / Examples.rows()).x(learningRate));
//             current = cost(theta, Examples, classifications);

//             i++;

//             if(last) {
//             if(current < last)
//                 learningRateFound = true;
//             else
//                 break;

//             if(last - current < 0.0001)
//                 break;
//             }

//             if(i >= maxIt) {
//                 throw 'unable to find minimum';
//             }

//             last = current;
//         }

//         learningRate /= 3;
//     }

//     return theta.chomp(1);
// }

var LogisticRegressionWorkersClassifier = function () {
  Classifier.call(this);
  this.examples = {};
  this.features = [];
  this.featurePositions = {};
  this.maxFeaturePosition = 0;
  this.classifications = [];
  this.exampleCount = 0;
};

util.inherits(LogisticRegressionWorkersClassifier, Classifier);

function createClassifications() {
  var classifications = [];

  for (var i = 0; i < this.exampleCount; i++) {
    var classification = [];

    for (var _ in this.examples) {
      classification.push(0);
    }

    classifications.push(classification);
  }

  return classifications;
}

async function computeThetasOld(Examples, Classifications) {
  this.theta = [];

  const numWorkers = 4;
  const chunkSize = Math.ceil(this.classifications.length / numWorkers);
  const chunks = [];
  for (let i = 0; i < this.classifications.length; i += chunkSize) {
    const chunk = this.classifications.slice(i, i + chunkSize);
    chunks.push({
      Examples,
      Classifications,
      startIndex: i,
      classifications: chunk,
    });
  }

  let completedWorkers = 0;

  return new Promise((resolve, reject) => {
    for (let i = 0; i < numWorkers; i++) {
      const worker = new Worker(
        path.resolve(__dirname, "descend_gradient.js"),
        {
          workerData: chunks[i],
        }
      );

      worker.on("message", (message) => {
        this.theta.push(...message);
      });

      worker.on("error", (error) => {
        reject(error);
        console.error(error);
      });

      worker.on("exit", () => {
        completedWorkers++;

        if (completedWorkers === numWorkers) {
          resolve();
        }
      });
    }
  });

  // each class will have it's own theta.
  //   var zero = function () {
  //     return 0;
  //   };
  //   for (var i = 1; i <= this.classifications.length; i++) {
  //     var theta = Examples.row(1).map(zero);
  //     this.theta.push(
  //       descendGradient(theta, Examples, Classifications.column(i))
  //     );
  //   }
}

async function computeThetas(examples, allClassifications) {
  this.theta = [];

  const numWorkers = 4;
  const chunkSize = Math.ceil(this.classifications.length / numWorkers);
  const chunks = [];
  for (let i = 0; i < this.classifications.length; i += chunkSize) {
    const chunk = this.classifications.slice(i, i + chunkSize);
    chunks.push({
      examples,
      allClassifications,
      startIndex: i,
      classifications: chunk,
    });
  }

  let completedWorkers = 0;

  return new Promise((resolve, reject) => {
    for (let i = 0; i < numWorkers; i++) {
      const worker = new Worker(
        path.resolve(__dirname, "descend_gradient.js"),
        {
          workerData: chunks[i],
        }
      );

      worker.on("message", (message) => {
        this.theta.push(...message);
      });

      worker.on("error", (error) => {
        reject(error);
        console.error(error);
      });

      worker.on("exit", () => {
        completedWorkers++;

        if (completedWorkers === numWorkers) {
          resolve();
        }
      });
    }
  });

  // each class will have it's own theta.
  //   var zero = function () {
  //     return 0;
  //   };
  //   for (var i = 1; i <= this.classifications.length; i++) {
  //     var theta = Examples.row(1).map(zero);
  //     this.theta.push(
  //       descendGradient(theta, Examples, Classifications.column(i))
  //     );
  //   }
}

async function train() {
  var examples = [];
  var classifications = this.createClassifications();
  var d = 0,
    c = 0;

  for (var classification in this.examples) {
    for (var i = 0; i < this.examples[classification].length; i++) {
      var doc = this.examples[classification][i];
      var example = doc;

      examples.push(example);
      classifications[d][c] = 1;
      d++;
    }

    c++;
  }

  //   await this.computeThetas($M(examples), $M(classifications));
  await this.computeThetas(examples, classifications);
}

function addExample(data, classification) {
  if (!this.examples[classification]) {
    this.examples[classification] = [];
    this.classifications.push(classification);
  }

  this.examples[classification].push(data);
  this.exampleCount++;
}

function getClassifications(observation) {
  observation = $V(observation);
  var classifications = [];

  for (var i = 0; i < this.theta.length; i++) {
    classifications.push({
      label: this.classifications[i],
      value: sigmoid(observation.dot(this.theta[i])),
    });
  }

  return classifications.sort(function (x, y) {
    return y.value - x.value;
  });
}

function restore(classifier) {
  classifier = Classifier.restore(classifier);
  classifier.__proto__ = LogisticRegressionWorkersClassifier.prototype;

  return classifier;
}

LogisticRegressionWorkersClassifier.prototype.addExample = addExample;
LogisticRegressionWorkersClassifier.prototype.restore = restore;
LogisticRegressionWorkersClassifier.prototype.train = train;
LogisticRegressionWorkersClassifier.prototype.createClassifications =
  createClassifications;
LogisticRegressionWorkersClassifier.prototype.computeThetas = computeThetas;
LogisticRegressionWorkersClassifier.prototype.getClassifications =
  getClassifications;

LogisticRegressionWorkersClassifier.restore = restore;

module.exports = LogisticRegressionWorkersClassifier;
