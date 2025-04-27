# MoE Lens - An Expert Is All You Need

## 📄 - [Press Here](https://openreview.net/forum?id=GS4WXncwSF) 

### [Marmik Chaudhari](https://www.marmik.xyz/)$^1$, [Idhant Gulati](https://www.idhant.xyz/)$^1$, [Nishkal Hundia](https://www.linkedin.com/in/nishkal-hundia/)$2$, [Pranav Karra](https://pranavkarra.me/)$^1$, & [Shivam Raval](https://shivamraval.scholars.harvard.edu/)$^3$

$^1$ The Pennsylvania State University, University Park, PA 16802 USA \
$^2$ University of Maryland, College Park, MD 20742 USA \
$^3$ Harvard University, Cambridge, MA 02138 USA

## Abstract
Mixture of Experts (MoE) models enable parameter-efficient scaling through sparse expert activations, yet optimizing their inference and memory costs remains challenging due to limited understanding of their specialization behavior. We present a systematic analysis of expert specialization in MoEs through two complementary approaches: domain-specific routing patterns and an early decoding framework that tracks expert contributions to output representations. Our analysis of the DeepSeekMoE model reveals that despite having 64 routed experts with 6 active for each layer's computation, the model predominantly relies on a few specialized experts, with the top-weighted expert's output closely approximating the full ensemble prediction. We quantitatively validate these findings through a systematic analysis of the token routing distribution, demonstrating that very few experts handle over 50\% of routing decisions across different specialized domains. Hidden state similarity between single and ensemble experts for every layer is extremely high, with some layers having cosine similarity as high as 0.95 and perplexity increasing by only 5\% when using a single expert across all three domains. Our results indicate that Mixture of Experts models exhibit concentrated expertise highlighting potential opportunities for inference optimization through targeted expert pruning while maintaining model performance and opening avenues towards studying localization of learned knowledge in these models.