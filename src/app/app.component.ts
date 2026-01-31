import { Component } from '@angular/core';
import { RouterOutlet } from '@angular/router';
import {NetworkComponent} from './components/network/network.component';

@Component({
  selector: 'app-root',
  imports: [RouterOutlet, NetworkComponent],
  templateUrl: './app.component.html',
  styleUrl: './app.component.css'
})
export class AppComponent {
  private title = 'cnn-visualizer';
}
