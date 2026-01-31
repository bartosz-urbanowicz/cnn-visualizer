import {multiply, add} from "mathjs";
import {sigmoid} from "../misc/activation-functions";
import {Dataset, Group} from "h5wasm";
import {map} from "mathjs";
import {TrainableLayer} from './trainable-layer';
import {DenseLayerParameters} from '../types/DenseLayerParameters';
import {Conv2dLayerParameters} from '../types/Conv2dLayerParameters';

export class Dense extends TrainableLayer {
	public inputShape: number = 0;
	public outputShape: number = 0;
	public parameters: DenseLayerParameters = {weights: [], biases: []};
	public lastActivations: number[] = [];
	public lastPreActivations: number[] = [];
	private input: number[] = [];

	public constructor(outputShape: number, activation: string) {
		super(activation);

		this.outputShape = outputShape;
	}

	public initialize(previousShape: number): void {
		this.inputShape = previousShape
		let limit = null;

		if (this.initializer === "xavier") {
			limit = Math.sqrt(6 / (previousShape + this.outputShape));
		} else if (this.initializer === "he") {
			limit = Math.sqrt(6 / previousShape);
		} else {
			throw new Error("select valid initializer");
		}

		this.parameters.weights = Array.from(
			{length: this.outputShape},
			() => Array.from({length: previousShape}, () => Math.random() * (2 * limit) - limit)
		);

		this.parameters.biases = Array.from({length: this.outputShape}, () => 0)
	}

	public importKerasWeights(data: Group, previousShape: number): void {
		const weightDataset: Dataset = data.get("vars/0") as Dataset
		const weights: number[] = Array.from(weightDataset.value as Int32Array)
		const biases: Dataset = data.get("vars/1") as Dataset

		this.inputShape = previousShape;

		const newWeights: number[][] = []

		for (let i = 0; i < this.outputShape; i++) {
			newWeights.push([])
			for (let j = 0; j < previousShape; j++) {
				newWeights[i].push(weights[(j * this.outputShape) + i]);
			}
		}

		this.parameters.weights = newWeights;
		this.parameters.biases = Array.from(biases.value as Int32Array)
	}

	public initializeGradient(): DenseLayerParameters {
		return {
			weights: Array.from({length: (this.outputShape as number)},
				() => Array((this.inputShape as number)).fill(0)),
			biases: Array((this.outputShape as number)).fill(0)
		}
	}

	public accumulateGradient(
		acc: DenseLayerParameters,
		gradient: DenseLayerParameters
	): DenseLayerParameters {
		for (let j = 0; j < gradient.weights.length; j++) {
			for (let i = 0; i < gradient.weights[j].length; i++) {
				acc.weights[j][i] += gradient.weights[j][i]
			}
			acc.biases[j] += acc.biases[j]
		}
		return acc;
	}

	public averageGradient(
		gradient: DenseLayerParameters,
		batchSize: number
	): DenseLayerParameters {
		for (let j = 0; j < gradient.weights.length; j++) {
			for (let i = 0; i < gradient.weights[j].length; i++) {
				gradient.weights[j][i] /= batchSize
			}
			gradient.biases[j] /= batchSize
		}
		return gradient;
	}

	public forward(input: number[]): number[] {
		// console.log("dense forward")
		const preActivations: number[] = add(multiply(this.parameters.weights, input), this.parameters.biases);
		const activations = this.activationFunction(preActivations);

		// stored for backprop
		this.lastPreActivations = preActivations;
		this.lastActivations = activations;
		this.input = input;

		return activations;
	}

	// this takes index instead of the value like in conv2d
	public activationFunctionDerivative(idx: number): number {
		if (this.activationFunctionName === "sigmoid") {
			const activation = this.lastActivations[idx]
			return activation * (1 - activation)
		}
		if (this.activationFunctionName === "relu") {
			const preActivation = this.lastPreActivations[idx]
			return preActivation > 0 ? 1 : 0;
		}
		return 0
	}

	public outputLayerWeightsGradient(losses: number[]): number[][] {
		const gradient: number[][] = []
		const deltas = []
		for (let j = 0; j < this.outputShape; j++) {
			gradient.push([])
			let delta = null;
			if (this.activationFunctionName === "sigmoid") {
				delta = losses[j] * this.activationFunctionDerivative(j);
			} else {
				delta = losses[j];
			}
			deltas.push(delta)
			for (let i = 0; i < this.inputShape; i++) {
				const changeToWeight = delta * this.input[i];
				gradient[j].push(changeToWeight)
			}
		}

		this.deltas = deltas;
		return gradient
	}

	public outputLayerBiasesGradient(losses: number[]): number[] {
		const gradient: number[] = []
		for (let j = 0; j < this.outputShape; j++) {
			let changeToBias = null;
			if (this.activationFunctionName === "sigmoid") {
				changeToBias = losses[j] * this.activationFunctionDerivative(j)
			} else {
				changeToBias = losses[j];
			}

			gradient.push(changeToBias)
		}

		return gradient
	}

	public backward(gradientWrtOutput: number[]): {gradientWrtInput: number[], weightsGradient: number[][], biasesGradient: number[]} {
		const weightsGradient: number[][] = []
		const biasesGradient: number[] = []
		const deltas = []

		for (let j = 0; j < this.outputShape; j++) {
			weightsGradient.push([])
			const delta = gradientWrtOutput[j] * this.activationFunctionDerivative(j)
			deltas.push(delta)
			biasesGradient.push(delta)
			for (let i = 0; i < this.inputShape; i++) {
				const changeToWeight = delta * this.input[i];
				weightsGradient[j].push(changeToWeight)
			}
		}


		let gradientWrtInput = [];
		for (let i = 0; i < this.inputShape; i++) {
			let sum = 0;
			for (let j = 0; j < this.parameters.weights.length; j++) {
				sum += (this.parameters.weights[j][i] as number) * deltas[j];
			}
			gradientWrtInput.push(sum);
		}

		return {gradientWrtInput: gradientWrtInput, biasesGradient: biasesGradient, weightsGradient: weightsGradient};
	}
}
