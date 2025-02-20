import pyautogui
from screeninfo import get_monitors

def take_screenshot(region_width=600, region_height=120, x_offset=250, y_offset=0):
    monitors = get_monitors()
    monitor = monitors[0]
    
    center_x = monitor.width//2
    center_y = monitor.height//2
    
    x = center_x - region_width//2 + x_offset + monitor.x
    y = center_y - region_height//2 + y_offset + monitor.y
    
    return pyautogui.screenshot(region=(x, y, region_width, region_height))