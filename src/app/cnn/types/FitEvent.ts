export type TFitEvent =
	| { type: "epochStart"; epoch: number; totalEpochs: number }
	| { type: "batchStart"; batch: number; totalBatches: number }
	| { type: "epochEnd"; epoch: number; valAccuracy: number }
