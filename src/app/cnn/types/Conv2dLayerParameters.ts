export interface Conv2dLayerParameters {
	weights: number[][][][], // filters, channels, kernel height, kernel width
	biases: number[] // one bias for each filter
}
