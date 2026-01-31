import {Layer} from "./layer.js";

export class Flatten extends Layer {

	public inputShape: [number, number, number] = [0, 0, 0];
	public outputShape: number = 0;

	public constructor() {
		super();
	}

	public initialize(previousShape: [number, number, number]): void {
		this.inputShape = previousShape;
		this.outputShape = previousShape[0] * previousShape[1] * previousShape[2];
	}


	// tensorflow is channel-last!
	public forward(input: number[][][]): number[] {
		const result: number[] = [];
		for (let channel = 0; channel < input.length; channel++) {
			for (let i = 0; i < input[channel].length; i++) {
				result.push(...input[channel][i])
			}
		}

		// console.log("flatten forward")

		return result;
	}

	public backward(gradientWrtOutput: number[]): number[][][] {
		const result: number[][][] = [];
		const stack = gradientWrtOutput.reverse();
		for (let c = 0; c < this.inputShape[0]; c++) {
			result.push([])
			for (let i = 0; i < this.inputShape[1]; i++) {
				result[c].push([])
				for (let j = 0; j < this.inputShape[2]; j++) {
					result[c][i].push(stack.pop()!);
				}
			}
		}

		return result;
	}
}
