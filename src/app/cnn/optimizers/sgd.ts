import {Network} from '../network';
import {LayerSgdMomentumState} from '../types/LayerSgdMomentumState';
import {Optimizer} from './optimizer';
import {Layer} from '../layers/layer';
import {TrainableLayer} from '../layers/trainable-layer';
import {Convolutional2d} from '../layers/convolutional-2d';
import {Dense} from '../layers/dense';

export class Sgd extends Optimizer {

	public state: {} = {};

	public constructor(learningRate: number) {
		super(learningRate);
	}

	public initializeStates(network: Network): void {
	};

	public applyGradient(layer: TrainableLayer, weightsGradient: number[][] | number[][][][], biasesGradient: number[]): void {

		if (layer instanceof Dense) {
			weightsGradient as number[][]
			for (let j = 0; j < layer.outputShape; j++) {
				for (let i = 0; i < layer.inputShape; i++) {
					const changeToWeight = this.learningRate * (weightsGradient as number[][])[j][i];
					layer.parameters.weights[j][i] -= changeToWeight;
				}
			}
		}

		if (layer instanceof Convolutional2d) {
			for (let filter = 0; filter < layer.filters; filter++) {
				for (let channel = 0; channel < layer.inputShape[0]; channel++) {
					for (let i = 0; i < layer.kernelSize[0]; i++) {
						for (let j = 0; j < layer.kernelSize[1]; j++) {
							const changeToWeight =
								this.learningRate * (weightsGradient as number[][][][])[filter][channel][j][i];
							layer.parameters.weights[filter][channel][j][i] -= changeToWeight;
						}
					}
				}
			}
		}

		for (let i = 0; i < layer.parameters.biases.length; i++) {
			const changeToBias = this.learningRate * biasesGradient[i];
			layer.parameters.biases[i] -= changeToBias;
		}


	}

}
