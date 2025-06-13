from pydantic import BaseModel, Field
from typing import Optional

class WeatherRecord(BaseModel):
    date: int = Field(..., description="Date as integer (e.g., YYYYMMDD)")
    cloud_cover: Optional[float] = Field(None, description="Cloud cover in oktas")
    sunshine: Optional[float] = Field(None, description="Sunshine duration in hours")
    global_radiation: Optional[float] = Field(None, description="Global radiation in MJ/m^2")
    max_temp: Optional[float] = Field(None, description="Maximum temperature in Celsius")
    mean_temp: Optional[float] = Field(None, description="Mean temperature in Celsius")
    min_temp: Optional[float] = Field(None, description="Minimum temperature in Celsius")
    precipitation: Optional[float] = Field(None, description="Precipitation in mm")
    pressure: Optional[float] = Field(None, description="Pressure in hPa")
    snow_depth: Optional[float] = Field(None, description="Snow depth in cm")
