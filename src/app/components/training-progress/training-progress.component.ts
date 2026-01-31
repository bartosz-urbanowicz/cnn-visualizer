import {Component, Input} from '@angular/core';
import { ProgressBarModule } from 'primeng/progressbar';

@Component({
  selector: 'app-training-progress',
  imports: [ProgressBarModule],
  templateUrl: './training-progress.component.html',
  styleUrl: './training-progress.component.css'
})
export class TrainingProgressComponent {

  @Input({ required: true }) public epochs!: number[];
  @Input({ required: true }) public numBatches!: number;
  @Input({ required: true }) public currentBatch!: number;
  @Input({ required: true }) public currentEpoch!: number;

  protected readonly NaN = NaN;
  protected readonly isNaN = isNaN;
}
