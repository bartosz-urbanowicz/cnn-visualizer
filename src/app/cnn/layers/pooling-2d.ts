import {Layer} from "./layer.js";
import {Group} from "h5wasm";
import {TTensorShape} from '../types/TensorShape';
import {multiply} from 'mathjs';
import {getShape} from '../misc/util';

export class Pooling2d extends Layer {

	public inputShape: [number, number, number] = [0, 0, 0];
	public outputShape: [number, number, number] = [0, 0, 0];
	private mask: number[][][] = [];
	private poolSize: [number, number]; // height, width
	private stride: number;
	private method: string;

	public constructor(poolSize: [number, number], stride: number, method: string) {
		super();

		this.poolSize = poolSize;
		this.stride = stride;
		this.method = method;
	}

	public initialize(previousShape: [number, number, number]):void {
		this.inputShape = previousShape;

		const heightOut = Math.floor((previousShape[1] - this.poolSize[0]) / this.stride) + 1;
		const widthOut = Math.floor((previousShape[2] - this.poolSize[1]) / this.stride) + 1;

		this.outputShape = [previousShape[0], heightOut, widthOut];
	}

	private maximum(image: number[][], vertOffset: number, horOffset: number): [number, [number, number]] {
		let max = -Infinity;
		let maxPosition: [number, number] = [NaN, NaN];
		for (let i = 0; i < this.poolSize[0]; i++) {
			for (let j = 0; j < this.poolSize[1]; j++) {
				const currValue = image[i + vertOffset][j + horOffset];
				if (currValue > max) {
					max = currValue;
					maxPosition = [i, j];
				}
			}
		}
		return [max, maxPosition];
	}

	private average(image: number[][], vertOffset: number, horOffset: number): number {
		let sum = 0;
		for (let i = 0; i < this.poolSize[0]; i++) {
			for (let j = 0; j < this.poolSize[1]; j++) {
				const currValue = image[i + vertOffset][j + horOffset];
				sum += currValue;
			}
		}
		return sum / (this.poolSize[0] * this.poolSize[1])
	}

	private initializeMask(): void {
		this.mask = Array.from({ length: this.inputShape[0] }, () =>
			Array.from({ length: this.inputShape[1] }, () =>
				Array.from({ length: this.inputShape[2] }, () => 0)
			)
		);
	}

	public slide(image: number[][], channel: number): number[][] {
		const result: number[][] = [[]];
		let currVertOffset = 0;
		let currHorOffset = 0;

		while (
			currHorOffset <= (image[0].length - this.poolSize[1]) &&
			currVertOffset <= (image.length - this.poolSize[0])
			) {
			if (this.method === "max") {
				const [max, maxPosition] = this.maximum(image, currVertOffset, currHorOffset);
				this.mask[channel][maxPosition[0] + currVertOffset][maxPosition[1] + currHorOffset] = 1;
				result[result.length - 1].push(max)
			} else if (this.method === "avg") {
				result[result.length - 1].push(this.average(image, currVertOffset, currHorOffset))
			}

			// console.log(currHorOffset, currVertOffset, result)

			// edge reached
			if (currHorOffset + this.stride + this.poolSize[1] > image[0].length) {
				currHorOffset = 0;
				currVertOffset += this.stride;
				if (currVertOffset + this.poolSize[0] <= image.length) {
					result.push([]);
				}
			} else {
				currHorOffset += this.stride;
			}
		}

		return result;
	}

	public forward(input: number[][][]): number[][][] {
		this.initializeMask();
		const result: number[][][] = [];
		for (let channel = 0; channel < input.length; channel++) {
			const image = input[channel];
			const channelResult = this.slide(image, channel);
			result.push(channelResult)
		}

		// console.log("pooling forward")

		return result;
	}

	public backward(gradientWrtOutput: number[][][]): number[][][] {
		const result: number[][][] = [...this.mask]

		// console.log(getShape(gradientWrtOutput))
		// console.log(this.inputShape[0], this.outputShape[1], this.outputShape[2])

		for (let i = 0; i < gradientWrtOutput[0].length; i++) {
			for (let j = 0; j < gradientWrtOutput[0][0].length; j++) {
				const heightFrom = i * this.stride;
				const heightTo = heightFrom + this.poolSize[0];
				const widthFrom = j * this.stride;
				const widthTo = widthFrom + this.poolSize[1];

				for (let c = 0; c < this.inputShape[0]; c++) {
					for (let h = heightFrom; h < heightTo; h++) {
						for (let w = widthFrom; w < widthTo; w++) {
							result[c][h][w] = this.mask[c][h][w] * gradientWrtOutput[c][i][j];
						}
					}
				}
			}

		}

		return result

	}
}
