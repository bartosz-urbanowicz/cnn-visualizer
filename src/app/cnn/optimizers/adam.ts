import {Network} from '../network';
import {LayerSgdMomentumState} from '../types/LayerSgdMomentumState';
import {Optimizer} from './optimizer';
import {Layer} from '../layers/layer';
import {LayerAdamState} from '../types/LayerAdamState';
import {TLayerOptimizerState} from '../types/LayerOptimizerState';

export class Adam extends Optimizer{

  private firstMomentDecay: number;
  private secondMomentDecay: number;
  private stabilizer: number;
  public state: { layers: LayerAdamState[], timestep: number } = { layers: [], timestep: 1 };

  public constructor(
    learningRate: number,
    firstMomentDecay: number = 0.9,
    secondMomentDecay: number = 0.999,
    stabilizer: number = 0.00000001
  ) {
    super(learningRate);
    this.firstMomentDecay = firstMomentDecay;
    this.secondMomentDecay = secondMomentDecay;
    this.stabilizer = stabilizer;
  }

  public initializeStates(network: Network): void {
    network.layers.forEach((layer) => {
      const layerState: LayerAdamState = {firstMoment: {weights: [], biases: []}, secondMoment: {weights: [], biases: []}};
      layerState.firstMoment.weights = Array.from(
        { length: layer.outputShape },
        () => Array.from({ length: layer.inputShape }, () => 0)
      );
      layerState.secondMoment.weights = Array.from(
        { length: layer.outputShape },
        () => Array.from({ length: layer.inputShape }, () => 0)
      );

      layerState.firstMoment.biases = Array.from({ length: layer.outputShape }, () => 0)
      layerState.secondMoment.biases = Array.from({ length: layer.outputShape }, () => 0)

      this.state.layers.push(layerState);
    })
    this.state.timestep = 0;
  };

  public applyGradient(layer: Layer, weightsGradient: number[][], biasesGradient: number[], layerIndex: number): void {
    for (let j = 0; j < layer.outputShape; j++) {
      for (let i = 0; i < layer.inputShape; i++) {
        const previousFirstMoment: number = this.state.layers[layerIndex].firstMoment.weights[j][i]
        const newFirstMoment: number =
          (this.firstMomentDecay * previousFirstMoment) + ((1 - this.firstMomentDecay) * weightsGradient[j][i]);
        this.state.layers[layerIndex].firstMoment.weights[j][i] = newFirstMoment;
        const firstMomentCorrected: number = newFirstMoment / (1 - Math.pow(this.firstMomentDecay, this.state.timestep))

        const previousSecondMoment: number =  this.state.layers[layerIndex].secondMoment.weights[j][i]
        const newSecondMoment: number =
          (this.secondMomentDecay * previousSecondMoment) + ((1 - this.secondMomentDecay) * Math.pow(weightsGradient[j][i], 2))
        this.state.layers[layerIndex].secondMoment.weights[j][i] = newSecondMoment;
        const secondMomentCorrected: number = newSecondMoment / (1 - Math.pow(this.secondMomentDecay, this.state.timestep))

        const changeToWeight: number =
          (this.learningRate * firstMomentCorrected) / (Math.sqrt(secondMomentCorrected) + this.stabilizer)
        layer.weights[j][i] -= changeToWeight;
      }
    }

    for (let i = 0; i < layer.biases.length; i++) {
      const previousFirstMoment: number = this.state.layers[layerIndex].firstMoment.biases[i]
      const newFirstMoment: number =
        (this.firstMomentDecay * previousFirstMoment) + ((1 - this.firstMomentDecay) * biasesGradient[i]);
      this.state.layers[layerIndex].firstMoment.biases[i] = newFirstMoment;
      const firstMomentCorrected: number = newFirstMoment / (1 - Math.pow(this.firstMomentDecay, this.state.timestep))

      const previousSecondMoment: number =  this.state.layers[layerIndex].secondMoment.biases[i]
      const newSecondMoment: number =
        (this.secondMomentDecay * previousSecondMoment) + ((1 - this.secondMomentDecay) * Math.pow(biasesGradient[i], 2))
      this.state.layers[layerIndex].secondMoment.biases[i] = newSecondMoment;
      const secondMomentCorrected: number = newSecondMoment / (1 - Math.pow(this.secondMomentDecay, this.state.timestep))

      const changeToBias: number =
        (this.learningRate * firstMomentCorrected) / (Math.sqrt(secondMomentCorrected) + this.stabilizer)
      layer.biases[i] -= changeToBias;
    }

  }
}
