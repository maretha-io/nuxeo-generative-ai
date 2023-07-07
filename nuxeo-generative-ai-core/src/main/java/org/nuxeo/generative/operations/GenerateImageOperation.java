/* Copyright 2023 Maretha Solutions LLC - https://maretha.io.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * 
 */
package org.nuxeo.generative.operations;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.json.JSONException;
import org.json.JSONObject;
import org.nuxeo.ecm.automation.core.annotations.Operation;
import org.nuxeo.ecm.automation.core.annotations.OperationMethod;
import org.nuxeo.ecm.automation.core.annotations.Param;
import org.nuxeo.ecm.automation.server.jaxrs.batch.Batch;
import org.nuxeo.ecm.automation.server.jaxrs.batch.BatchManager;
import org.nuxeo.ecm.core.api.Blob;
import org.nuxeo.ecm.core.api.impl.blob.StringBlob;
import org.nuxeo.generative.ai.GenerativeAI;
import org.nuxeo.generative.ai.GenerativeAIProvider;
import org.nuxeo.generative.commons.ProjectConstant;
import org.nuxeo.runtime.api.Framework;

import java.io.IOException;
import java.util.Arrays;
import java.util.Base64;
import java.util.stream.Collectors;

@Operation(id = GenerateImageOperation.ID, category = ProjectConstant.CAT_GENERATIVE_AI, label = "Generate an image", description = "Generate an image based on given conditions")
public class GenerateImageOperation {

    public static final String ID = "GenerativeAi.GenerateImage";

    private static final Log log = LogFactory.getLog(GenerateImageOperation.class);

    @Param(name = "prompt")
    protected String prompt;

    @Param(name = "style", required = false)
    protected String[] style;

    @OperationMethod
    public Blob run() {
        try {
            Blob result = Framework.getService(GenerativeAI.class)
                                   .generateImage(null, combineParameters(prompt, style),
                                           GenerativeAIProvider.IMG_1024);
            if (result != null) {
                BatchManager bm = Framework.getService(BatchManager.class);
                String batchId = bm.initBatch();
                Batch batch = bm.getBatch(batchId);
                batch.addFile("0", result, result.getFilename(), result.getMimeType());
                return new StringBlob(toJson(result.getFilename(), batchId, result).toString(), "application/json");
            }
        } catch (Exception e) {
            log.error(e);
            return new StringBlob(messageToJson(e.getMessage()).toString(), "application/json");
        }
        return new StringBlob(messageToJson("No image generated, please try again").toString(), "application/json");
    }

    private String combineParameters(String prompt, String[] style) {
        // Combine the parameters into a single string
        StringBuilder combined = new StringBuilder();
        if (prompt != null) {
            combined.append(prompt);
        }
        if (style != null) {
			var styles = Arrays.stream(style).map(x -> x + " style").collect(Collectors.joining(" and "));
            combined.append(" in a ").append(styles);
        }
        return combined.toString().trim();
    }

    private JSONObject toJson(String title, String batchId, Blob image) throws JSONException, IOException {
        JSONObject json = new JSONObject();
        json.put("title", title);
        json.put("batchId", batchId);
        json.put("content", Base64.getEncoder().encodeToString(image.getByteArray()));
        return json;
    }

    private JSONObject messageToJson(String message) throws JSONException {
        JSONObject json = new JSONObject();
        json.put("message", message);
        return json;
    }
}
