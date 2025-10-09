# Copyright (C) 2025 Xiaomi Corporation.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

import dasheng
import torch
import torch.nn as nn


class MusicInstrumentRecognize_Dasheng(nn.Module):

    def __init__(
        self,
        hidden_size: list[int] = [256, 128, 64],
        fine_tune: bool = False,
        use_dropout: bool = False,
        use_batchnorm: bool = False,
        **kwargs,
    ):
        super(MusicInstrumentRecognize_Dasheng, self).__init__()

        # Dasheng audio encoder
        self.dashengmodel = dasheng.dasheng_base()
        self.num_embeddings = self.dashengmodel.embed_dim

        # Load pre-trained weights
        state_dict = torch.load("./dasheng_audioset_mAP497.pt", map_location="cpu")
        self.dashengmodel.load_state_dict(state_dict, strict=False)

        self.fine_tune = fine_tune

        if fine_tune == False:
            # Freeze all encoder parameters
            for param in self.dashengmodel.parameters():
                param.requires_grad = False
        else:
            # Unfreeze all encoder parameters
            for param in self.dashengmodel.parameters():
                param.requires_grad = True

        # Build classifier head
        self.binary_classifier = nn.Sequential()
        prev_dim = self.num_embeddings
        for idx, dim in enumerate(hidden_size):
            # Linear layer
            self.binary_classifier.add_module(f"linear_{idx}", nn.Linear(prev_dim, dim))

            # BatchNorm layer (optional)
            if use_batchnorm:
                self.binary_classifier.add_module(f"bn_{idx}", nn.BatchNorm1d(dim))

            # Activation function
            self.binary_classifier.add_module(f"relu_{idx}", nn.ReLU())

            # Dropout layer (optional)
            if use_dropout:
                self.binary_classifier.add_module(f"dropout_{idx}", nn.Dropout(0.3))
            prev_dim = dim
        
        # Final output layer
        final_layer = nn.Linear(prev_dim, 1)
        self.binary_classifier.add_module("final_output", final_layer)

        self.loss_fn = nn.BCEWithLogitsLoss()  # Sigmoid + BCE


    def forward(self, x: torch.Tensor):
        # Normalize input waveform
        epsilon = 1e-9
        x = x / (x.abs().amax(dim=1, keepdim=True) + epsilon)

        # Forward pass through encoder (with or without gradients)
        with torch.no_grad() if not self.fine_tune else torch.enable_grad():
            x = self.dashengmodel(x)  # [batch, 72, 768]

        # Average pooling over time dimension
        x = x.mean(dim=1)  # [batch, 768]

        # Classification head
        x = self.binary_classifier(x)  # [batch, 1]
        x = x.squeeze(1)  # Remove extra dimension
        return x
