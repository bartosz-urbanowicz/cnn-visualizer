import {TFitEvent} from '../types/FitEvent';

export function getShape(arr: any) {
	const shape = [];
	let current = arr;
	while (Array.isArray(current)) {
		shape.push(current.length);
		current = current[0];
	}
	return shape;
}

export function logFitEvent(event: TFitEvent): void {
	switch (event.type) {
		case "epochStart":
			console.log(`Epoch ${event.epoch}/${event.totalEpochs}`);
			break;

		case "batchStart":
			// replacing the last line
			// process.stdout.write(`batch ${event.batch}/${event.totalBatches}\r`)
      console.log((`batch ${event.batch}/${event.totalBatches}\r`))
			break;

		case "epochEnd":
			console.log(`val_accuracy: ${event.valAccuracy}`)
			break;
	}
}
