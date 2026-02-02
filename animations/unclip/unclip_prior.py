from manim import *
import numpy as np

class unCLIPPrior(Scene):
    def construct(self):
        # Colors
        text_color = "#FF6B6B"
        clip_text_color = "#4ECDC4"
        timestep_color = "#FFE66D"
        image_color = "#95E1D3"
        final_color = "#C7CEEA"
        output_color = "#6C5CE7"
        
        # Position for sequence building (will grow to the right)
        sequence_start_pos = LEFT * 5 + UP * 0.5
        current_x = sequence_start_pos[0]
        
        # === STEP 1: Text to Token Embeddings ===
        step1_label = Text("Step 1: Tokenize and Embed Text", 
                          font_size=18, color=text_color).to_edge(DOWN, buff=0.3)
        self.play(Write(step1_label))
        
        # Original text
        text_input = Text('"a sweet bunny"', font_size=24)
        text_input.shift(LEFT * 5 + UP * 2.5)
        self.play(FadeIn(text_input))
        self.wait(0.5)
        
        # Token embeddings
        tokens = ["a", "sweet", "bunny"]
        token_embeds = VGroup()
        for i, token in enumerate(tokens):
            embed = self.create_embedding_box(token, text_color, width=0.7, height=1.2)
            embed.shift(np.array([current_x + i * 0.85, 0.5, 0]))
            token_embeds.add(embed)
        
        arrow1 = Arrow(text_input.get_bottom(), token_embeds.get_top(), 
                      color=text_color, buff=0.2)
        self.play(Create(arrow1), FadeIn(token_embeds))
        self.wait(1)
        
        current_x += len(tokens) * 0.85
        
        # === STEP 2: Text to CLIP Embedding ===
        self.play(FadeOut(step1_label))
        step2_label = Text("Step 2: Generate CLIP Text Embedding", 
                          font_size=18, color=clip_text_color).to_edge(DOWN, buff=0.3)
        self.play(Write(step2_label))
        
        # CLIP encoder box
        clip_encoder = RoundedRectangle(width=2, height=1, corner_radius=0.2,
                                       color=clip_text_color, fill_opacity=0.2)
        clip_encoder.shift(LEFT * 1 + UP * 2.5)
        clip_label = Text("CLIP Text\nEncoder", font_size=14, color=clip_text_color)
        clip_label.move_to(clip_encoder.get_center())
        clip_encoder_group = VGroup(clip_encoder, clip_label)
        
        arrow2 = Arrow(text_input.get_right(), clip_encoder.get_left(),
                      color=clip_text_color, buff=0.1)
        
        self.play(Create(arrow2), FadeIn(clip_encoder_group))
        self.wait(0.5)
        
        # CLIP text embedding output
        clip_text_embed = self.create_embedding_box("CLIP\nText", clip_text_color, 
                                                    width=0.9, height=1.2)
        clip_text_embed.shift(np.array([current_x + 0.5, 0.5, 0]))
        
        arrow3 = Arrow(clip_encoder.get_bottom(), clip_text_embed.get_top(),
                      color=clip_text_color, buff=0.2)
        self.play(Create(arrow3), FadeIn(clip_text_embed))
        self.wait(1)
        
        current_x += 1.05
        
        # === STEP 3: Concatenate Token + CLIP Embeddings ===
        self.play(FadeOut(step2_label))
        step3_label = Text("Step 3: Concatenate Token Embeddings + CLIP Embedding", 
                          font_size=18, color=WHITE).to_edge(DOWN, buff=0.3)
        self.play(Write(step3_label))
        
        # Fade out text and arrows
        self.play(
            FadeOut(text_input),
            FadeOut(arrow1),
            FadeOut(arrow2),
            FadeOut(arrow3),
            FadeOut(clip_encoder_group)
        )
        
        # Highlight concatenation
        concat_bracket1 = Brace(VGroup(token_embeds, clip_text_embed), DOWN, color=WHITE)
        concat_text1 = Text("Text Sequence", font_size=14, color=WHITE)
        concat_text1.next_to(concat_bracket1, DOWN, buff=0.1)
        self.play(GrowFromCenter(concat_bracket1), Write(concat_text1))
        self.wait(1)
        self.play(FadeOut(concat_bracket1), FadeOut(concat_text1))
        
        # === STEP 4: Add Timestep Encoding ===
        self.play(FadeOut(step3_label))
        step4_label = Text("Step 4: Add Timestep Encoding", 
                          font_size=18, color=timestep_color).to_edge(DOWN, buff=0.3)
        self.play(Write(step4_label))
        
        # Timestep value
        t_value = MathTex("t = 5", font_size=24, color=timestep_color)
        t_value.shift(UP * 2.5 + RIGHT * 2)
        self.play(FadeIn(t_value))
        self.wait(0.3)
        
        # Timestep embedding
        timestep_embed = self.create_embedding_box("t=5", timestep_color, 
                                                   width=0.7, height=1.2)
        timestep_embed.shift(np.array([current_x + 0.4, 0.5, 0]))
        
        arrow4 = Arrow(t_value.get_bottom(), timestep_embed.get_top(),
                      color=timestep_color, buff=0.2)
        self.play(Create(arrow4), FadeIn(timestep_embed))
        self.wait(1)
        self.play(FadeOut(t_value), FadeOut(arrow4))
        
        current_x += 0.85
        
        # === STEP 5: Image to Noised CLIP Embedding ===
        self.play(FadeOut(step4_label))
        step5_label = Text("Step 5: Add Noise to Image → CLIP Image Embedding", 
                          font_size=18, color=image_color).to_edge(DOWN, buff=0.3)
        self.play(Write(step5_label))
        
        # Image
        image = ImageMobject("media/images/unclip_prior/image_0.JPEG").scale(0.3)
        image_box = SurroundingRectangle(image, color=WHITE, buff=0)
        image_group = Group(image_box, image)
        image_group.shift(RIGHT * 4 + UP * 3)
        self.play(FadeIn(image_group))
        self.wait(1)
        
        # Create a random array the same size as the image to simulate noise
        noise = np.random.normal(0, 0.5, image.get_pixel_array().shape)
        noisy_image_array = np.clip(image.get_pixel_array() + noise, 0, 1)
        noisy_image = ImageMobject(noisy_image_array)
        noisy_image_group = Group(noisy_image)
        noisy_image_group.shift(RIGHT * 4 + UP * 3)
        
        noise_label = Text("+ Noise", font_size=14, color=RED)
        noise_label.next_to(image_group, RIGHT, buff=0.2)
        
        self.play(
            FadeIn(noisy_image_group),
            Write(noise_label)
        )
        self.wait(1)
        
        # CLIP Image Encoder
        clip_img_encoder = RoundedRectangle(width=2, height=1, corner_radius=0.2,
                                           color=image_color, fill_opacity=0.2)
        clip_img_encoder.shift(RIGHT * 4)
        clip_img_label = Text("CLIP Image\nEncoder", font_size=14, color=image_color)
        clip_img_label.move_to(clip_img_encoder.get_center())
        clip_img_encoder_group = VGroup(clip_img_encoder, clip_img_label)
        
        arrow5 = Arrow(image_group.get_bottom(), clip_img_encoder.get_top(),
                      color=image_color, buff=0.2)
        self.play(Create(arrow5), FadeIn(clip_img_encoder_group))
        self.wait(1)
        
        # Noised CLIP image embedding
        noised_img_embed = self.create_embedding_box("Noised\nImg", image_color,
                                                     width=0.9, height=1.2)
        noised_img_embed.shift(np.array([current_x + 0.5, 0.5, 0]))
        
        arrow6 = Arrow(clip_img_encoder.get_left(), noised_img_embed.get_right(),
                      color=image_color, buff=0.2)
        self.play(Create(arrow6), FadeIn(noised_img_embed))
        self.wait(1)
        
        self.play(
            FadeOut(image_group),
            FadeOut(noise_label),
            FadeOut(arrow5),
            FadeOut(arrow6),
            FadeOut(clip_img_encoder_group)
        )
        
        current_x += 1.05
        
        # === STEP 6: Add Final Prediction Token ===
        self.play(FadeOut(step5_label))
        step6_label = Text("Step 6: Add Final Prediction Token", 
                          font_size=18, color=final_color).to_edge(DOWN, buff=0.3)
        self.play(Write(step6_label))
        
        final_embed = self.create_embedding_box("[?]", final_color,
                                                width=0.7, height=1.2)
        final_embed.shift(np.array([current_x + 0.4, 0.5, 0]))
        
        self.play(FadeIn(final_embed))
        self.wait(1)
        
        current_x += 0.85
        
        # Show full sequence
        full_sequence = VGroup(token_embeds, clip_text_embed, timestep_embed, 
                              noised_img_embed, final_embed)
        
        # Bracket showing full input
        full_bracket = Brace(full_sequence, DOWN, color=YELLOW)
        full_label = Text("Complete Input Sequence", font_size=16, 
                         color=YELLOW, weight=BOLD)
        full_label.next_to(full_bracket, DOWN, buff=0.1)
        self.play(GrowFromCenter(full_bracket), Write(full_label))
        self.wait(1)
        
        # === STEP 7: Process Through Transformer ===
        self.play(FadeOut(step6_label))
        step7_label = Text("Step 7: Process Through Decoder-Only Transformer", 
                          font_size=18, color=BLUE).to_edge(DOWN, buff=0.3)
        self.play(Write(step7_label))
        
        # Move sequence up and show transformer
        self.play(
            full_sequence.animate.shift(UP * 1.5),
            FadeOut(full_bracket),
            FadeOut(full_label)
        )
        
        # Transformer
        transformer = self.create_transformer()
        transformer.shift(DOWN * 1)
        
        arrow_down = Arrow(full_sequence.get_bottom(), transformer.get_top(),
                          color=WHITE, buff=0.2)
        self.play(Create(arrow_down), FadeIn(transformer))
        self.wait(1)
        
        # Highlight final token position
        highlight = final_embed.copy().set_color(YELLOW)
        self.play(
            Indicate(final_embed, scale_factor=1.3, color=YELLOW),
            run_time=1
        )
        
        # Output prediction
        output_embed = self.create_embedding_box("Clean\nImage\nEmb", output_color,
                                                 width=1.2, height=1.2)
        output_embed.shift(DOWN * 3)
        
        arrow_out = Arrow(transformer.get_bottom(), output_embed.get_top(),
                         color=output_color, buff=0.2)
        output_label = Text("Predicted Unnoised CLIP Image Embedding",
                           font_size=14, color=output_color, weight=BOLD)
        output_label.next_to(output_embed, DOWN, buff=0.2)
        
        self.play(FadeOut(step7_label), Create(arrow_out), FadeIn(output_embed), Write(output_label))
        self.wait(2)
        
    def create_embedding_box(self, label, color, width=0.8, height=1.0):
        """Create an embedding box with label"""
        box = Rectangle(width=width, height=height,
                       color=color, fill_opacity=0.3, stroke_width=3)
        text = Text(label, font_size=12, color=color, weight=BOLD)
        text.move_to(box.get_center())
        return VGroup(box, text)
    
    def create_transformer(self):
        """Create transformer block visualization"""
        # Main box
        outer = RoundedRectangle(width=6, height=1.5, corner_radius=0.2,
                                color=BLUE, stroke_width=3, fill_opacity=0.1)
        
        # Internal layers
        layers = VGroup()
        layer_names = ["Causal Self-Attention", "Feed-Forward", "Layer Norm"]
        for i, name in enumerate(layer_names):
            layer = Rectangle(width=5.5, height=0.35,
                            color=BLUE_C, fill_opacity=0.2)
            layer.shift(UP * 0.4 - DOWN * i * 0.4)
            layer_label = Text(name, font_size=10, color=BLUE_C)
            layer_label.move_to(layer.get_center())
            layers.add(VGroup(layer, layer_label))

        layers.move_to(outer.get_center())
        
        # Title
        title = Text("Decoder-Only Transformer", font_size=14, 
                    color=BLUE, weight=BOLD)
        title.next_to(outer, UP, buff=0.1)
        
        return VGroup(outer, layers, title)