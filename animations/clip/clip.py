from manim import *
import numpy as np

class CLIPDemo(Scene):
    def construct(self):
        # Define colors for encoders
        image_color = BLUE
        text_color = GREEN
        
        # 1. Create and display input image (top left)
        # Load image from file
        image = ImageMobject("media/images/clip/image_0.JPEG").scale(0.5)
        # Make the image box outline the image
        image_box = SurroundingRectangle(image, color=WHITE, buff=0)
        image_group = Group(image_box, image)
        image_group.shift(LEFT*4 + UP*1.5)
        image_label = Text("Input Image", font_size=20).next_to(image_group, UP, buff=0.2)
        
        # Fade in image
        self.play(FadeIn(image_group), FadeIn(image_label))
        self.wait(1)
        
        # 2. Create and display text description (below image)
        text_desc = Text('"A couple of bubbas."', font_size=24)
        text_box = SurroundingRectangle(text_desc, color=WHITE, buff=0.3)
        text_group = VGroup(text_box, text_desc)
        text_group.next_to(image_group, 2*DOWN, buff=1)
        text_label = Text("Text Description", font_size=20).next_to(text_group, UP, buff=0.2)
        
        # Fade in text
        self.play(FadeIn(text_group), FadeIn(text_label))
        self.wait(1)
        
        # 3. Create encoders
        # Image Encoder
        image_encoder = VGroup(
            RoundedRectangle(
                width=2, height=1.5, 
                corner_radius=0.2,
                color=image_color,
                fill_opacity=0.3
            ),
            Tex("$f_{image}$", font_size=35, color=image_color)
        )
        image_encoder.move_to(image_group.get_center() + RIGHT*4)
        
        # Text Encoder
        text_encoder = VGroup(
            RoundedRectangle(
                width=2, height=1.5,
                corner_radius=0.2,
                color=text_color,
                fill_opacity=0.3
            ),
            Tex("$f_{text}$", font_size=35, color=text_color)
        )
        text_encoder.move_to(text_group.get_center() + RIGHT*4)
        
        # Show encoders
        self.play(
            GrowFromCenter(image_encoder),
            GrowFromCenter(text_encoder)
        )
        self.wait(1)
        
        # Add arrows from inputs to encoders
        arrow_img = Arrow(
            image_group.get_right(), 
            image_encoder.get_left(),
            color=image_color,
            buff=0.1
        )
        arrow_text = Arrow(
            text_group.get_right(),
            text_encoder.get_left(),
            color=text_color,
            buff=0.1
        )
        
        self.play(
            Create(arrow_img),
            Create(arrow_text)
        )
        self.wait(1)
        
        # 4. Create 2D embedding plot on the right
        # Set up axes
        axes = Axes(
            x_range=[-1, 1, 0.5],
            y_range=[-1, 1, 0.5],
            x_length=3,
            y_length=3,
            axis_config={"color": GREY},
            tips=False
        ).shift(RIGHT*3.5)
        
        axes_labels = axes.get_axis_labels(x_label="d_1", y_label="d_2")
        plot_title = Text("Embedding Space", font_size=24).next_to(axes, 2*UP, buff=0.3)
        
        self.play(
            Create(axes),
            Write(axes_labels),
            Write(plot_title)
        )
        self.wait(0.5)
        
        # Create embedding vectors
        # Image embedding vector
        img_embedding = np.array([0.6, 0.7])
        img_vector = Arrow(
            axes.c2p(0, 0),
            axes.c2p(img_embedding[0], img_embedding[1]),
            color=image_color,
            buff=0,
            stroke_width=6,
            max_tip_length_to_length_ratio=0.15
        )
        
        # Text embedding vector (close to image vector to show similarity)
        text_embedding = np.array([0.55, 0.75])
        text_vector = Arrow(
            axes.c2p(0, 0),
            axes.c2p(text_embedding[0], text_embedding[1]),
            color=text_color,
            buff=0,
            stroke_width=6,
            max_tip_length_to_length_ratio=0.15
        )
        
        # Animate vectors appearing from encoders
        arrow_to_plot_img = Arrow(
            image_encoder.get_right(),
            axes.c2p(0, 0),
            color=image_color,
            stroke_opacity=0.3
        )
        arrow_to_plot_text = Arrow(
            text_encoder.get_right(),
            axes.c2p(0, 0),
            color=text_color,
            stroke_opacity=0.3
        )
        
        self.play(
            Create(arrow_to_plot_img),
            Create(arrow_to_plot_text)
        )
        
        self.play(
            Create(img_vector),
            Create(text_vector),
        )
        
        # Final pause
        self.wait(5)