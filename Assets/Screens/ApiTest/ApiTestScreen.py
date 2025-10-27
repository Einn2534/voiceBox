# Created on: 2025-10-28
# Author: nullptr
# Description: Placeholder API test screen implementation.

from kivy.lang import Builder
from kivy.uix.screenmanager import Screen

Builder.load_string(
    """
<ApiTestLayout@BoxLayout>:
    orientation: "vertical"
    spacing: 16
    padding: 24

<ApiTestScreen>:
    ApiTestLayout:
        Label:
            text: "API Test Screen"
            font_size: "20sp"
        Button:
            text: "戻る"
            on_release: app.root.current = "home"
    """
)


class ApiTestScreen(Screen):
    """Provide a placeholder screen for API connectivity checks."""

    def on_enter(self):
        """Log that the API test screen became visible."""
        print("ApiTest Screen Entered!")
