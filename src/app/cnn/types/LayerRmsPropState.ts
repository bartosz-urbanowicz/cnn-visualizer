import {DenseLayerParameters} from './DenseLayerParameters';
import {Conv2dLayerParameters} from './Conv2dLayerParameters';

export interface LayerRmsPropState {
  avgSquareGradient: DenseLayerParameters | Conv2dLayerParameters;
}
