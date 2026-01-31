import {LayerSgdMomentumState} from './LayerSgdMomentumState';
import {LayerAdamState} from './LayerAdamState';
import {LayerRmsPropState} from './LayerRmsPropState';

export type TLayerOptimizerState = LayerAdamState | LayerSgdMomentumState | LayerRmsPropState | {};
