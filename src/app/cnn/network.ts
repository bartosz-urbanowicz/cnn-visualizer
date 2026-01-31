import {Layer} from "./layers/layer";
import {Dense} from "./layers/dense"
import {Group} from "h5wasm";
import {DenseLayerParameters} from './types/DenseLayerParameters';
import {Sample} from './types/Sample';
import {NetworkParams} from './types/NetworkParams';
import {accuracy} from './misc/metrics';
import {Optimizer} from './optimizers/optimizer';
import {TrainableLayer} from './layers/trainable-layer';
import {TTensorShape} from './types/TensorShape';
import {Conv2dLayerParameters} from './types/Conv2dLayerParameters';
import { Observable, Subscriber } from "rxjs";
import {TFitEvent} from './types/FitEvent';

export class Network {
	public layers: Layer[]
	public trainableLayers: TrainableLayer[]
	private optimizer: Optimizer;

	public constructor(
		layers: Layer[],
		optimizer: Optimizer
	) {
		this.layers = layers
		// this.trainableLayers = layers.slice(1, layers.length) as TrainableLayer[]
		this.trainableLayers = [...layers].filter((layer) => {
			return layer instanceof TrainableLayer
		});

		this.trainableLayers.forEach((layer: TrainableLayer) => {
			layer.optimizer = optimizer;
		})

		let previousLayer: Layer | null = null
		this.layers.forEach((layer, index) => {
			layer.setPreviousLayer(previousLayer)
			if (index != layers.length - 1) {
				layer.setNextLayer(layers[index + 1])
			}
			previousLayer = layer;
		})

		this.optimizer = optimizer;
	}

	public async importKerasWeights(file: string, inputShape: number): Promise<void> {
		const h5wasm = await import("h5wasm/node");
		await h5wasm.ready;

		let f = new h5wasm.File(file, "r");

		const layers: Group = f.get("layers")! as Group


		let previousShape: TTensorShape = inputShape
		// input layers weights are not imported, but it has to be initialized
		this.layers[0].initialize(previousShape)

		// TODO refactor for trainable layers only
		// layers.keys().forEach((layer: string, i: number) => {
		// 	const currentLayer: Layer = this.layers[i + 1]
		// 	currentLayer.importKerasWeights(layers.get(layer)! as Group, previousShape)
		// 	previousShape = currentLayer.outputShape
		// })
	}

	public initialize(): void {
		let previousShape = this.layers[0].inputShape
		this.layers.forEach(layer => {
			layer.initialize(previousShape);
			previousShape = layer.outputShape;
		})
	}

	public predict(input: number[] | number[][][]): number[] {
		let output: number[] | number[][][] = input;
		this.layers.forEach(layer => {
			output = layer.forward(output);
		})
		// dense layer is always last
		return output as number[]
	}

	public gradient(batch: Sample[]): (DenseLayerParameters | Conv2dLayerParameters)[] {
		const gradientSum: (DenseLayerParameters | Conv2dLayerParameters)[] =
			this.trainableLayers.map((layer: TrainableLayer) => {
				return layer.initializeGradient();
			});

		for (const sample of batch) {
			const prediction: number[] = this.predict(sample.data)
			const losses: number[] = prediction.map((pred, idx) => {
				return pred - sample.target[idx]
			})

			// output layers
			const outputLayer = this.trainableLayers[this.trainableLayers.length - 1];

			const outputLayerGradients: DenseLayerParameters = {
				weights: (outputLayer as Dense).outputLayerWeightsGradient(losses),
				biases: (outputLayer as Dense).outputLayerBiasesGradient(losses)
			}

			let outputLayerOutputGradient = [];
			// dense layer is always last
			for (let i = 0; i < (outputLayer as Dense).inputShape; i++) {
				let sum = 0;
				for (let j = 0; j < outputLayer.parameters.weights.length; j++) {
					sum += (outputLayer.parameters.weights[j][i] as number) * outputLayer.deltas[j];
				}
				outputLayerOutputGradient.push(sum);
			}

			// hidden layers
			const hiddenLayersGradients: DenseLayerParameters[] = []

			let outputGradient = outputLayerOutputGradient;
			// omit input and output layer
			this.layers.slice(1, -1).reverse().forEach((layer) => {
				if (layer instanceof TrainableLayer) {
					const result = layer.backward(outputGradient);
					const weightsGradient = result["weightsGradient"];
					const biasesGradient = result["biasesGradient"];
					outputGradient = result["gradientWrtInput"]
					hiddenLayersGradients.push({
						weights: weightsGradient,
						biases: biasesGradient
					})
				} else {
					outputGradient = layer.backward(outputGradient);
				}

			})

			const singleGradient: (DenseLayerParameters | Conv2dLayerParameters)[] = [
				...hiddenLayersGradients.reverse(),
				outputLayerGradients
			]

			singleGradient.forEach((layerGradient: DenseLayerParameters | Conv2dLayerParameters, idx: number) => {
				gradientSum[idx] = this.trainableLayers[idx].accumulateGradient(
					gradientSum[idx],
					layerGradient
				);
			})
		}

		// avg
		gradientSum.forEach((layerGradient, idx) => {
			gradientSum[idx] =
				this.trainableLayers[idx].averageGradient(layerGradient, batch.length);
		})

		return gradientSum;

	}

	public applyGradient(gradient: (DenseLayerParameters | Conv2dLayerParameters)[]): void {
		// only for adam
		// this.trainableLayers[0].optimizer.state.timestep! += 1;
		this.trainableLayers.forEach((layer: TrainableLayer, idx) => {
			const weightsGradient = gradient[idx].weights
			const biasesGradient = gradient[idx].biases
			layer.optimizer!.applyGradient(layer, weightsGradient, biasesGradient, idx)
		})
	}

	public printWeights(): void {
		this.trainableLayers.forEach((layer, i) => {
			console.log(`Layer ${i + 1}:`)
			console.log("weights:")
			console.log(layer.parameters.weights)
			console.log("biases:")
			console.log(layer.parameters.biases)
			console.log("")
		})
	}

	public runEpoch(
		trainingData: Sample[],
		batchSize: number,
		subscriber: Subscriber<TFitEvent>
	): void {

		// fisher yates shuffle
		const shuffled = trainingData.slice();
		for (let i = shuffled.length - 1; i > 0; i--) {
			const j = Math.floor(Math.random() * (i + 1));
			[shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
		}

		const numBatches: number = Math.ceil(trainingData.length / batchSize)

		for (let i = 0; i < numBatches; i++) {

			subscriber.next({
				type: "batchStart",
				batch: i,
				totalBatches: numBatches,
			});

			const start = i * batchSize;
			const end = Math.min(start + batchSize, shuffled.length);
			const batch = shuffled.slice(start, end);
			const gradient: (DenseLayerParameters | Conv2dLayerParameters)[] = this.gradient(batch);
			this.applyGradient(gradient);
		}
	}

	public fit(
		inputs: number[][] | number[][][][],
		outputs: number[][],
		params: NetworkParams
	): Observable<TFitEvent> {
		return new Observable<TFitEvent>((subscriber: Subscriber<TFitEvent>) => {
			(async () => {
				this.optimizer.initializeStates(this);

				const data: Sample[] = inputs.map((input, index) => ({
					data: input,
					target: outputs[index]
				}));

				const shuffled = data.slice();
				for (let i = shuffled.length - 1; i > 0; i--) {
					const j = Math.floor(Math.random() * (i + 1));
					[shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
				}

				const splitIndex = Math.floor(data.length * params.validationSplit);
				const trainingData = data.slice(0, splitIndex);
				const validationData = data.slice(splitIndex);

				for (let i = 0; i < params.epochs; i++) {
					// console.log(`Epoch ${i + 1}/${params.epochs}`)
					subscriber.next({ type: "epochStart", epoch: i, totalEpochs: params.epochs });

					this.runEpoch(trainingData, params.batchSize, subscriber);

					const acc: number = accuracy(this, validationData)
					// console.log(`val_accuracy: ${acc}`)
					subscriber.next({ type: "epochEnd", epoch: i, valAccuracy: acc });
				}

				subscriber.complete()

			})().catch(err => subscriber.error(err));
		})
	}
}
