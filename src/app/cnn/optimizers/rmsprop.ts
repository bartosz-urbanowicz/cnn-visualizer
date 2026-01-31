import {Network} from '../network';
import {Optimizer} from './optimizer';
import {LayerRmsPropState} from '../types/LayerRmsPropState';
import {TrainableLayer} from '../layers/trainable-layer';
import {Dense} from '../layers/dense';
import {Convolutional2d} from '../layers/convolutional-2d';

export class RmsProp extends Optimizer {
	public state: { layers: LayerRmsPropState[] } = {layers: []};
	private decayRate: number;
	private stabilizer: number;

	public constructor(
		learningRate: number,
		decayRate: number = 0.9,
		stabilizer: number = 0.00000001
	) {
		super(learningRate);
		this.decayRate = decayRate;
		this.stabilizer = stabilizer;
	}

	public initializeStates(network: Network): void {
		network.trainableLayers.forEach((layer) => {
			const layerState: LayerRmsPropState = {avgSquareGradient: {weights: [], biases: []}};

			if (layer instanceof Dense) {
				layerState.avgSquareGradient.weights = Array.from(
					{length: layer.outputShape},
					() => Array.from({length: layer.inputShape}, () => 0)
				);
				layerState.avgSquareGradient.biases = Array.from({length: layer.outputShape}, () => 0)
			}

			if (layer instanceof Convolutional2d) {
				layerState.avgSquareGradient.weights = Array.from(
					{ length: layer.filters },
					() => Array.from(
						{ length: layer.inputShape[0] },
						() => Array.from(
							{ length: layer.kernelSize[0] },
							() => Array.from({ length: layer.kernelSize[1] }, () => 0)
						)
					)
				);
				layerState.avgSquareGradient.biases = Array.from({length: layer.filters}, () => 0);
			}

			this.state.layers.push(layerState);
		})
	};

	public applyGradient(
		layer: TrainableLayer,
	 	weightsGradient: number[][] | number[][][][],
		biasesGradient: number[],
		layerIndex: number
	): void {
		if (layer instanceof Dense) {
			for (let j = 0; j < layer.outputShape; j++) {
				for (let i = 0; i < layer.inputShape; i++) {
					const previousAvgSquareGradient: number =
						this.state.layers[layerIndex].avgSquareGradient.weights[j][i] as number
					const newAvgSquareGradient =
						this.decayRate *
						previousAvgSquareGradient + ((1 - this.decayRate) *
						Math.pow(weightsGradient[j][i] as number, 2))
					const changeToWeight =
						(this.learningRate *
						(weightsGradient[j][i] as number)) /
						(Math.sqrt(newAvgSquareGradient) + this.stabilizer)
					layer.parameters.weights[j][i] -= changeToWeight;
					this.state.layers[layerIndex].avgSquareGradient.weights[j][i] = newAvgSquareGradient;
				}
			}
		}

		if (layer instanceof Convolutional2d) {
			weightsGradient as number[][][][];
			for (let f = 0; f < layer.filters; f++) {
				for (let c = 0; c < layer.inputShape[0]; c++) {
					for (let i = 0; i < layer.kernelSize[0]; i++) {
						for (let j = 0; j < layer.kernelSize[1]; j++) {
							const prevAvg =
								(this.state.layers[layerIndex].avgSquareGradient.weights[f][c] as number[][])[i][j];
							const newAvg = this.decayRate *
								prevAvg + (1 - this.decayRate) *
								Math.pow((weightsGradient as number[][][][])[f][c][i][j], 2);
							const change = (this.learningRate *
								(weightsGradient as number[][][][])[f][c][i][j]) /
								(Math.sqrt(newAvg) + this.stabilizer);
							layer.parameters.weights[f][c][i][j] -= change;
							(this.state.layers[layerIndex].avgSquareGradient.weights[f][c] as number[][])[i][j] = newAvg;
						}
					}
				}
			}
		}

		for (let i = 0; i < layer.parameters.biases.length; i++) {
			const previousAvgSquareGradient = this.state.layers[layerIndex].avgSquareGradient.biases[i]
			const newAvgSquareGradient =
				this.decayRate * previousAvgSquareGradient + ((1 - this.decayRate) * Math.pow(biasesGradient[i], 2))
			const changeToBias =
				(this.learningRate * biasesGradient[i]) / (Math.sqrt(newAvgSquareGradient) + this.stabilizer)
			layer.parameters.biases[i] -= changeToBias;
			this.state.layers[layerIndex].avgSquareGradient.biases[i] = newAvgSquareGradient;
		}
	}


}
