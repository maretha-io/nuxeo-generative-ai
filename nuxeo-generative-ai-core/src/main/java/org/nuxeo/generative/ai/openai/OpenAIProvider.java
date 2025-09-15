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
 */
package org.nuxeo.generative.ai.openai;

import java.io.IOException;
import java.util.Base64;
import java.util.List;
import java.util.Map;
import java.util.Optional;

import org.apache.commons.lang3.StringUtils;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.nuxeo.ecm.core.api.Blob;
import org.nuxeo.ecm.core.api.Blobs;
import org.nuxeo.ecm.core.api.NuxeoException;
import org.nuxeo.generative.ai.GenerativeAIProvider;
import org.nuxeo.generative.ai.GenerativeAIProviderDescriptor;
import org.nuxeo.runtime.api.Framework;

import com.openai.client.OpenAIClient;
import com.openai.client.okhttp.OpenAIOkHttpClient;
import com.openai.models.images.Image;
import com.openai.models.images.ImageGenerateParams;
import com.openai.models.images.ImagesResponse;
import com.openai.models.images.ImageModel;

public class OpenAIProvider implements GenerativeAIProvider {

    private static final Log log = LogFactory.getLog(OpenAIProvider.class);

    public static final String ORGANIZATION_CONF_PARAM = "generative.ai.openai.organization";
    public static final String API_KEY_CONF_PARAM = "generative.ai.openai.apikey";

    // Environment variable fallbacks
    public static final String ENV_VAR_ORGANIZATION = "NUXEO_GENERATIVE_AI_OPENAI_ORGANIZATION";
    public static final String ENV_VAR_APIKEY = "NUXEO_GENERATIVE_AI_OPENAI_APIKEY";

    protected String organization;
    protected String apiKey;
    protected OpenAIClient client;
    protected GenerativeAIProviderDescriptor descriptor;

    public OpenAIProvider(GenerativeAIProviderDescriptor desc) {
        descriptor = desc;
        Map<String, String> params = desc.getParameters();

        if (params != null) {
            organization = params.get("organization");
            if (StringUtils.isBlank(organization)) {
                organization = Framework.getProperty(ORGANIZATION_CONF_PARAM);
            }
            if (StringUtils.isBlank(organization)) {
                // read from env (not system property)
                organization = System.getenv(ENV_VAR_ORGANIZATION);
            }

            apiKey = params.get("apiKey");
            if (StringUtils.isBlank(apiKey)) {
                apiKey = Framework.getProperty(API_KEY_CONF_PARAM);
            }
            if (StringUtils.isBlank(apiKey)) {
                // read from env (not system property)
                apiKey = System.getenv(ENV_VAR_APIKEY);
            }

            if (StringUtils.isBlank(apiKey)) {
                log.error("OpenAI API key is empty — service calls will be skipped.");
            } else {
                OpenAIOkHttpClient.Builder builder = OpenAIOkHttpClient.builder().apiKey(apiKey);
                if (StringUtils.isNotBlank(organization)) {
                    builder.organization(organization);
                }
                client = builder.build();
            }
        }
    }

    protected boolean checkApiKey() {
        if (StringUtils.isBlank(apiKey)) {
            log.warn("OpenAI API key is not configured — not calling the service.");
            return false;
        }
        return true;
    }

    @Override
    public Blob generateImage(String prompt, String size) throws IOException {
        if (!checkApiKey()) {
            log.error("No API key configured, returning null");
            return null;
        }

        try {
            // Build request for the latest image model (base64 output only for gpt-image-1)
            ImageGenerateParams.Builder builder = ImageGenerateParams.builder()
                .model(ImageModel.GPT_IMAGE_1)
                .prompt(prompt)
                // gpt-image-1 supports AUTO, 1024x1024, 1536x1024, 1024x1536
                .size(StringUtils.isBlank(size)
                        ? ImageGenerateParams.Size.AUTO
                        : ImageGenerateParams.Size.of(size))
                // gpt-image-1 uses base64; choose final file type via outputFormat:
                .outputFormat(ImageGenerateParams.OutputFormat.PNG)
                // background AUTO lets the model decide; TRANSPARENT requires PNG/WEBP
                .background(ImageGenerateParams.Background.AUTO);

            ImagesResponse response = client.images().generate(builder.build());

            Optional<List<Image>> imagesOpt = response.data();
            if (imagesOpt.isPresent() && !imagesOpt.get().isEmpty()) {
                Image img = imagesOpt.get().get(0);

                // gpt-image-1 returns base64; url() is generally not present for this model
                Optional<String> b64 = img.b64Json();
                if (b64.isPresent()) {
                    byte[] bytes = Base64.getDecoder().decode(b64.get());
                    Blob blob = Blobs.createBlob(bytes);
                    blob.setMimeType("image/png");
                    blob.setFilename("openai-image.png");
                    return blob;
                }

                // Fallback (for older models): URL download if provided
                Optional<String> url = img.url();
                if (url.isPresent()) {
                    return GenerativeAIProvider.downloadFile(url.get(), "", "OpenAI");
                }
            }

            log.warn("OpenAI returned no images for the given prompt.");
            return null;

        } catch (Exception e) {
            log.error("Cannot generate image with OpenAI", e);
            throw new NuxeoException(e);
        }
    }

    @Override
    public String getName() {
        return descriptor.getName();
    }
}
