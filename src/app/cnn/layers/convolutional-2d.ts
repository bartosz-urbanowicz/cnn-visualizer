import {Layer} from "./layer.js";
import {Group} from "h5wasm";
import {TrainableLayer} from './trainable-layer';
import {Conv2dLayerParameters} from '../types/Conv2dLayerParameters';
import {add, sum} from 'mathjs';
import {DenseLayerParameters} from '../types/DenseLayerParameters';

export class Convolutional2d extends TrainableLayer {
	public inputShape: [number, number, number] = [0, 0, 0]; // channels, height, width
	public outputShape: [number, number, number] = [0, 0, 0]; // channels, height, width
	public filters: number;
	public kernelSize: [number, number]; // height, width
	private padding: number;
	public parameters: Conv2dLayerParameters = { weights: [], biases: [] };
	private stride: number;
	public lastActivations: number[][][] = [];
	public lastPreActivations: number[][][] = [];
	private input: number[][][] = [];

	public constructor(
		activation: string,
		filters: number, // out channels
		kernelSize: [number, number],
		padding: number,
		stride: number
	) {
		super(activation);

		this.filters = filters;
		this.kernelSize = kernelSize;
		this.padding = padding;
		this.stride = stride;
	}

	public initialize(previousShape: [number, number, number]): void {
		this.inputShape = previousShape

		const heightOut = Math.floor((previousShape[1] + 2 * this.padding - this.kernelSize[0]) / this.stride) + 1;
		const widthOut = Math.floor((previousShape[2] + 2 * this.padding - this.kernelSize[1]) / this.stride) + 1;

		this.outputShape = [this.filters, heightOut, widthOut];

		let limit = null;
		if (this.initializer === "xavier") {
			limit = Math.sqrt(6 / (
				(previousShape[0] * this.kernelSize[0] * this.kernelSize[1]) +
				(this.filters * this.kernelSize[0] * this.kernelSize[1])
			));
		} else if (this.initializer === "he") {
			limit = Math.sqrt(6 / (previousShape[0] * this.kernelSize[0] * this.kernelSize[1]));
		} else {
			throw new Error("select valid initializer");
		}

		this.parameters.weights = Array.from(
			{length: this.filters},
			() => Array.from(
				{length: previousShape[0]},
				() => Array.from(
					{length: this.kernelSize[0]},
					() => Array.from(
						{length: this.kernelSize[1]},
						() => Math.random() * (2 * limit) - limit)
				)
			)
		);

		this.parameters.biases = Array.from({length: this.filters}, () => 0)
	}

	public activationFunctionDerivative(x: number): number {
		return x > 0 ? 1 : 0;
	}

	public importKerasWeights(data: Group, previousShape: number): void {
		throw new Error("Method not implemented.");
	}

	public initializeGradient(): Conv2dLayerParameters {
		return {
			weights: Array.from({ length: this.filters }, () =>
				Array.from({ length: this.inputShape[0] }, () =>
					Array.from({ length: this.inputShape[1] }, () =>
						Array(this.inputShape[2]).fill(0)
					)
				)
			),
			biases: Array(this.filters).fill(0)
		};
	}

	public accumulateGradient(
		acc: Conv2dLayerParameters,
		grad: Conv2dLayerParameters
	): Conv2dLayerParameters {
		for (let f = 0; f < grad.weights.length; f++) {
			for (let c = 0; c < grad.weights[f].length; c++) {
				for (let y = 0; y < grad.weights[f][c].length; y++) {
					for (let j = 0; j < grad.weights[f][c][y].length; j++) {
						acc.weights[f][c][y][j] += grad.weights[f][c][y][j];
					}
				}
			}
			acc.biases[f] += grad.biases[f];
		}
		return acc;
	}

	averageGradient(
		gradient: Conv2dLayerParameters,
		batchSize: number
	): Conv2dLayerParameters {
		for (let f = 0; f < gradient.weights.length; f++) {
			for (let c = 0; c < gradient.weights[f].length; c++) {
				for (let i = 0; i < gradient.weights[f][c].length; i++) {
					for (let j = 0; j < gradient.weights[f][c][i].length; j++) {
						gradient.weights[f][c][i][j] /= batchSize;
					}
				}
			}
			gradient.biases[f] /= batchSize;
		}

		return gradient;
	}

	public pad(image: number[][]): number[][] {
		const result: number[][] = [];
		// top
		for (let i = 0; i < this.padding; i++) {
			result.push(Array(image[0].length + 2 * this.padding).fill(0))
		}
		// center
		for (let i = 0; i < image.length; i++) {
			result.push(Array(this.padding).fill(0).concat(image[i], Array(this.padding).fill(0)))
		}
		// bottom
		for (let i = 0; i < this.padding; i++) {
			result.push(Array(image[0].length + 2 * this.padding).fill(0))
		}

		return result;
	}

	// does not work for backprop operations so for now replaced with correlate
	public slide(filter: number[][], image: number[][]): number[][] {
		const result: number[][] = [[]];
		let currVertOffset = 0;
		let currHorOffset = 0;

		while (
			true
			) {

			if (
				currVertOffset + this.kernelSize[0] > image.length ||
				currHorOffset + this.kernelSize[1] > image[0].length
			) {
				break;
			}

			let sum = 0;
			filter.forEach((row, i) => {
				row.forEach((value, j) => {
					const currValue = image[i + currVertOffset][j + currHorOffset];
					sum += currValue * value;
				})
			})
			result[result.length - 1].push(sum)

			// console.log(currHorOffset, currVertOffset, result)

			// edge reached
			if (currHorOffset + this.stride + this.kernelSize[1] > image[0].length) {
				currHorOffset = 0;
				currVertOffset += this.stride;
				if (currVertOffset + this.kernelSize[0] <= image.length) {
					result.push([]);
				}
			} else {
				currHorOffset += this.stride;
			}
		}

		return result;
	}

	public correlate(kernel: number[][], input: number[][]): number[][] {
		const kernelH = kernel.length;
		const kernelW = kernel[0].length;

		const outH =
			Math.floor((input.length - kernelH) / this.stride) + 1;
		const outW =
			Math.floor((input[0].length - kernelW) / this.stride) + 1;

		const result: number[][] = Array.from(
			{ length: outH },
			() => Array(outW).fill(0)
		);

		for (let i = 0; i < outH; i++) {
			for (let j = 0; j < outW; j++) {
				let sum = 0;

				for (let ki = 0; ki < kernelH; ki++) {
					for (let kj = 0; kj < kernelW; kj++) {
						sum +=
							input[i * this.stride + ki][j * this.stride + kj] *
							kernel[ki][kj];
					}
				}

				result[i][j] = sum;
			}
		}

		return result;
	}

	public forward(input: number[][][]): number[][][] {
		const preActivations: number[][][] = [];
		for (let filter = 0; filter < this.filters; filter++) {
			const kernel = this.parameters.weights[filter][0];
			const paddedImage = this.pad(input[0]);

			const outH = Math.floor(
				(paddedImage.length - kernel.length) / this.stride
			) + 1;

			const outW = Math.floor(
				(paddedImage[0].length - kernel[0].length) / this.stride
			) + 1;


			let filterResult: number[][] = Array.from(
				{ length: outH },
				() => Array(outW).fill(0)
			);

			for (let channel = 0; channel < input.length; channel++) {
				const image = this.pad(input[channel]);
				const channelResult = this.correlate(this.parameters.weights[filter][channel], image);
				filterResult = add(filterResult, channelResult);
			}
			const bias = this.parameters.biases[filter];
			for (let i = 0; i < filterResult.length; i++) {
				for (let j = 0; j < filterResult[0].length; j++) {
					filterResult[i][j] += bias;
				}
			}
			preActivations.push(filterResult);
		}

		// TODO allow other functions than relu+
		const activations: number[][][] = [];
		for (let i = 0; i < preActivations.length; i++) {
			activations.push([])
			for (let j = 0; j < preActivations[0].length; j++) {
				activations[activations.length - 1]
					.push(preActivations[i][j]
						.map((x) => Math.max(0, x)))
			}
		}

		// save for backprop
		this.lastPreActivations = preActivations;
		this.lastActivations = activations;
		this.input = input;

		// console.log("conv forward")

		return activations;
	}

	private flip(filter: number[][]): number[][] {
		return filter
			.slice()
			.reverse()
			.map(row => row.slice().reverse());
	}

	public backward(gradientWrtOutput: number[][][]):
		{gradientWrtInput: number[][][], weightsGradient: number[][][][], biasesGradient: number[]} {

		// biases gradient
		const deltas: number[][][] = []
		const biasesGradient: number[] = []

		for (let filter = 0; filter < this.outputShape[0]; filter++) {
			deltas.push([])
			let deltaSum = 0;
			for (let i = 0; i < this.outputShape[1]; i++) {
				deltas[filter].push([])
				for (let j = 0; j < this.outputShape[2]; j++) {
					const delta =
						gradientWrtOutput[filter][i][j] *
						this.activationFunctionDerivative(this.lastPreActivations[filter][i][j])
					deltas[filter][i].push(delta)
					deltaSum += delta;
				}
			}
			biasesGradient[filter] = deltaSum;
		}

		// weights gradient
		const weightsGradient: number[][][][] = []
		for (let filter = 0; filter < gradientWrtOutput.length; filter++) {
			weightsGradient.push([])
			for (let channel = 0; channel < this.input.length; channel++) {
				const input = this.pad(this.input[channel]);
				const channelResult = this.correlate(deltas[filter], input);
				weightsGradient[filter].push(channelResult)
			}
		}

		// gradient w.r.t to input
		const gradientWrtInput: number[][][][]  = []
		for (let filter = 0; filter < gradientWrtOutput.length; filter++) {
			gradientWrtInput.push([])
			for (let channel = 0; channel < this.input.length; channel++) {
				const paddedDeltas = this.pad(deltas[filter]);
				const flippedFilter = this.flip(this.parameters.weights[filter][channel]);
				const channelResult = this.correlate(flippedFilter, paddedDeltas);
				gradientWrtInput[filter].push(channelResult);
			}
		}

		// reduce to 3D
		const gradient3D: number[][][] = gradientWrtInput.reduce((acc, filterGrad) => {
			return acc.map((channel, cIdx) => channel.map((row, i) =>
				row.map((val, j) => val + filterGrad[cIdx][i][j])
			));
		});

		return {gradientWrtInput: gradient3D, biasesGradient: biasesGradient, weightsGradient: weightsGradient};
	}
}
