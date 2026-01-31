import {Component, OnInit} from '@angular/core';
import {Sample} from '../../cnn/types/Sample';
// @ts-ignore
import mnist from "mnist";
import {logFitEvent} from '../../cnn/misc/util';
import {Subscription} from 'rxjs';
import {TrainingProgressComponent} from '../training-progress/training-progress.component';
import {NetworkDetailsComponent} from '../network-details/network-details.component';

@Component({
  selector: 'app-network',
  imports: [TrainingProgressComponent, NetworkDetailsComponent],
  templateUrl: './network.component.html',
  styleUrl: './network.component.css'
})
export class NetworkComponent implements OnInit{

  private IMAGE_SIZE: number = 28;

  protected training: boolean = true;
  protected testAccuracy: number = 0;

  protected epochsNumber: number = 5;
  protected batchSize: number = 32;
  protected validationSplit: number = 0.8;

  protected currentEpoch: number = 0;
  protected epochs: number[] = [];
  protected numBatches: number = 0;
  protected currentBatch: number = 0;

  public constructor() {
    this.epochs = Array.from({ length: this.epochsNumber }, () => NaN)
  }

  private flatTo3D(flatArray: number[]): number[][][] {
    const image2D: number[][] = [];
    for (let i = 0; i < this.IMAGE_SIZE; i++) {
      image2D.push(flatArray.slice(i * this.IMAGE_SIZE, (i + 1) * this.IMAGE_SIZE));
    }
    return [image2D];
  }

  public ngOnInit(): void {
    const set = mnist.set(1200, 200);

    const trainingSet = set.training;
    const testSet = set.test;

    const testSet3D: Sample[] = testSet.map((sample: any) => {
      return {
        data: this.flatTo3D(sample.input),
        target: sample.output
      }
    })

    const inputsTrain: number[][] = trainingSet.map((item: { input: number[], output: number[] }) => item.input);
    const outputsTrain: number[][] = trainingSet.map((item: { input: number[], output: number[] }) => item.output);

    // const inputsTrain3D: number[][][][] = inputsTrain.map(this.flatTo3D);
    // so we dont lose this binding
    const inputsTrain3D: number[][][][] = inputsTrain.map(input => this.flatTo3D(input));

    const worker = new Worker(new URL('network.worker.ts', import.meta.url));
    const trainingParams = { epochs: this.epochsNumber, batchSize: this.batchSize, validationSplit: this.validationSplit };
    worker.postMessage({ inputsTrain3D, outputsTrain, testSet3D, trainingParams });

    worker.onmessage = ({ data }) => {
      if (data.type === 'progress') {
        switch (data.event.type) {
          case "epochStart":
            this.currentBatch = 0;
            break;
          case "batchStart":
            // TODO change so its not updated every batch
            this.numBatches = data.event.totalBatches;
            this.currentBatch += 1;
            break;
          case "epochEnd":
            this.epochs[this.currentEpoch] = data.event.valAccuracy;
            this.currentEpoch += 1;
            break;
        }
      } else if (data.type === 'complete') {
        this.training = false;
        this.testAccuracy = data.accuracy;
      } else if (data.type === 'error') {
        console.error('Training error:', data.error);
      }
    };
  }
}
