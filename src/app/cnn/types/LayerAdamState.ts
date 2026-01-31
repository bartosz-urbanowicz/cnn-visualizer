import {DenseLayerParameters} from './DenseLayerParameters';

export interface LayerAdamState {
  firstMoment: DenseLayerParameters;
  secondMoment: DenseLayerParameters;
}
