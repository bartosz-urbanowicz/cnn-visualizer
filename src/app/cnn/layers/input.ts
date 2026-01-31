import {Layer} from "./layer.js";
import {TTensorShape} from '../types/TensorShape';

export class Input extends Layer {

	public inputShape: TTensorShape;
	public outputShape: TTensorShape;

    public constructor(inputShape: TTensorShape) {
        super();

        this.inputShape = inputShape;
        this.outputShape = inputShape;
    }

    public initialize(previousShape: number):void {}

    public forward(input: number[]): number[] {
        return input;
    }

	public backward(gradientWrtOutput: any): any {
		throw new Error("Method not implemented");
	}
}
