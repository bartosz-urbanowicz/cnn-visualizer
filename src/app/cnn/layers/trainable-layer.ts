import {Group} from "h5wasm";
import {Optimizer} from '../optimizers/optimizer';
import {Layer} from './layer';
import {relu, sigmoid, softmax} from '../misc/activation-functions';
import {TTensorShape} from '../types/TensorShape';
import {Conv2dLayerParameters} from '../types/Conv2dLayerParameters';
import {DenseLayerParameters} from '../types/DenseLayerParameters';

export abstract class TrainableLayer extends Layer {
	public abstract parameters: DenseLayerParameters | Conv2dLayerParameters;
	protected activationFunction: (x: number[]) => number[];
	protected activationFunctionName: string;
	protected initializer: string = "";
	// can be undefined because added in network constructor for every layer
	public optimizer: Optimizer | undefined;
	public abstract lastActivations: number[] | number[][][];
	public abstract lastPreActivations: number[] | number[][][];

	public constructor(activation: string) {
		super();

		if (activation === "sigmoid") {
			this.activationFunction = sigmoid;
			this.activationFunctionName = "sigmoid";
			this.initializer = "xavier"
		} else if (activation === "softmax") {
			this.activationFunction = softmax;
			this.activationFunctionName = "softmax";
			this.initializer = "xavier"
		} else if (activation === "relu") {
			this.activationFunction = relu;
			this.activationFunctionName = "relu";
			this.initializer = "he"
		} else {
			throw new Error("Provide correct activation function!")
		}

	}

	public abstract importKerasWeights(data: Group, previousShape: TTensorShape): void;

	public abstract activationFunctionDerivative(x: number): number;

	public abstract initializeGradient(): DenseLayerParameters | Conv2dLayerParameters;

	public abstract accumulateGradient(
		acc: DenseLayerParameters | Conv2dLayerParameters,
		gradient: DenseLayerParameters | Conv2dLayerParameters
	): DenseLayerParameters | Conv2dLayerParameters;

	public abstract averageGradient(
		gradient: DenseLayerParameters | Conv2dLayerParameters,
		batchSize: number
	): DenseLayerParameters | Conv2dLayerParameters;
}
