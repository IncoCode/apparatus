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
const { spawn } = require("child_process");

var sylvester = require("sylvester"),
  Matrix = sylvester.Matrix,
  Vector = sylvester.Vector;

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

async function computeThetas(examples, allClassifications) {
  this.theta = new Array(this.classifications.length).fill(undefined);
  const data = this.theta.map((_, index) => ++index);

  const numWorkers = 4;
  const chunkSize = Math.ceil(data.length / numWorkers);
  const chunks = [];
  for (let i = 0; i < data.length; i += chunkSize) {
    const chunk = data.slice(i, i + chunkSize);
    chunks.push({
      examples,
      allClassifications,
      indexes: chunk,
    });
  }

  let completedWorkers = 0;

  const insertChunk = (chunk, index) => {
    for (let i = 0; i < chunk.length; i++) {
      this.theta[index + i] = chunk[i];
    }
  };

  return new Promise((resolve, reject) => {
    for (let i = 0; i < numWorkers; i++) {
      const childProcess = spawn(
        "node",
        path.resolve(__dirname, "descend_gradient2.js")
      );

      childProcess.stdin.write(JSON.stringify(chunks[i]));
      childProcess.stdin.end();

      childProcess.stdout.on("data", (data) => {
        // const result = data.toString();
        // Process the result
        // console.log(`Received result: ${result}`);

        const message = JSON.parse(data.toString());
        const index = message.startNumber - 1;
        insertChunk(message.thetas, index);
      });

      childProcess.on("exit", (code) => {
        console.log(`Child exited with code ${code}`);

        completedWorkers++;
        if (completedWorkers === numWorkers) {
          resolve();
        }
      });

      // const worker = new Worker(
      //   path.resolve(__dirname, "descend_gradient.js"),
      //   {
      //     workerData: chunks[i],
      //   }
      // );
      // worker.on("message", (message) => {
      //   const index = message.startNumber - 1;
      //   insertChunk(message.thetas, index);
      // });
      // worker.on("error", (error) => {
      //   reject(error);
      //   console.error(error);
      // });
      // worker.on("exit", () => {
      //   completedWorkers++;
      //   if (completedWorkers === numWorkers) {
      //     resolve();
      //   }
      // });
    }
  });
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
