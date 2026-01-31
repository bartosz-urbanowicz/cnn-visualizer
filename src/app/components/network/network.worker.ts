/// <reference lib="webworker" />

import {logFitEvent} from '../../cnn/misc/util';
import {Subscription} from 'rxjs';
import {Convolutional2d} from '../../cnn/layers/convolutional-2d';
import {Pooling2d} from '../../cnn/layers/pooling-2d';
import {Flatten} from '../../cnn/layers/flatten';
import {accuracy} from '../../cnn/misc/metrics';
import {RmsProp} from '../../cnn/optimizers/rmsprop';
import {Network} from '../../cnn/network';
import {Dense} from "../../cnn/layers/dense";
import {Input} from "../../cnn/layers/input";

addEventListener('message', ({ data }) => {
  const { inputsTrain3D, outputsTrain, testSet3D, trainingParams } = data;
  const { epochs, batchSize, validationSplit } = trainingParams;

  const model = new Network([
      new Input([1, 28, 28]),
      new Convolutional2d("relu", 8, [3, 3], 1, 1),
      new Pooling2d([2, 2], 2, "max"),
      new Convolutional2d("relu", 8, [3, 3], 1, 1),
      new Pooling2d([2, 2], 2, "max"),
      new Flatten(),
      new Dense(16, "relu"),
      new Dense(10, "softmax")
    ],
    // new Sgd(0.001)
    // new SgdMomentum(0.1)
    // best for mnist
    new RmsProp(0.001)
    // new Adam(0.001)
  )

  model.initialize()

  const subscription: Subscription = model.fit(
    inputsTrain3D,
    outputsTrain,
    {
      batchSize: batchSize,
      epochs: epochs,
      validationSplit: validationSplit
    }
  ).subscribe({
    next: event => postMessage({ type: 'progress', event }),
    complete: () => {
      const acc = accuracy(model, testSet3D);
      postMessage({ type: 'complete', accuracy: acc });
    },
    error: err => postMessage({ type: 'error', error: err.message })
  })

  subscription.unsubscribe();
});
