import {Network} from '../network';
import {LayerSgdMomentumState} from '../types/LayerSgdMomentumState';
import {Optimizer} from './optimizer';
import {Layer} from '../layers/layer';

export class SgdMomentum extends Optimizer{

  public state: { layers: LayerSgdMomentumState[] } = { layers: [] };
  private momentumCoefficient: number;

  public constructor(learningRate: number, momentumCoefficient: number = 0.9) {
    super(learningRate);
    this.momentumCoefficient = momentumCoefficient;
  }

  public initializeStates(network: Network): void {
    network.layers.forEach((layer) => {
      const layerState: LayerSgdMomentumState = {velocity: {weights: [], biases: []}};
      layerState.velocity.weights = Array.from(
        { length: layer.outputShape },
        () => Array.from({ length: layer.inputShape }, () => 0)
      );

      layerState.velocity.biases = Array.from({ length: layer.outputShape }, () => 0)

      this.state.layers.push(layerState);
    })
  };

  public applyGradient(layer: Layer, weightsGradient: number[][], biasesGradient: number[], layerIndex: number): void {
    for (let j = 0; j < layer.outputShape; j++) {
      for (let i = 0; i < layer.inputShape; i++) {
        const previousVelocity = this.state.layers[layerIndex].velocity.weights[j][i]
        const changeToWeight = (this.momentumCoefficient * previousVelocity) - (this.learningRate * weightsGradient[j][i])
        layer.weights[j][i] -= changeToWeight;
        this.state.layers[layerIndex].velocity.weights[j][i] = changeToWeight;
      }
    }

    for (let i = 0; i < layer.biases.length; i++) {
      const previousVelocity = this.state.layers[layerIndex].velocity.biases[i]
      const changeToBias = (this.momentumCoefficient * previousVelocity) - (this.learningRate * biasesGradient[i])
      layer.biases[i] -= changeToBias;
      this.state.layers[layerIndex].velocity.biases[i] = changeToBias;
    }
  }


}
