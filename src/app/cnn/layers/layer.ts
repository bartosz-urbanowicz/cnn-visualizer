import {Group} from "h5wasm";
import {Optimizer} from '../optimizers/optimizer';
import {TTensorShape} from '../types/TensorShape';
import {DenseLayerParameters} from '../types/DenseLayerParameters';
import {Conv2dLayerParameters} from '../types/Conv2dLayerParameters';

export abstract class Layer {
	public abstract inputShape: TTensorShape;
	public abstract outputShape: TTensorShape;
	protected previousLayer: null | Layer = null;
	protected nextLayer: null | Layer = null;
	public deltas: number[] = [];

	public abstract initialize(previousShape: TTensorShape): void

	public abstract forward(input: number[] | number[][][]): number[] | number[][][]

	// in non-trainable layers only gradient w.r.t input is returned
	public abstract backward(gradientWrtOutput: any):
		{gradientWrtInput: any, weightsGradient: any, biasesGradient: any} | any;

	public setPreviousLayer(layer: Layer | null): void {
		this.previousLayer = layer;
	}

	public setNextLayer(layer: Layer | null): void {
		this.nextLayer = layer;
	}
}
