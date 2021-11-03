from pydantic import BaseModel, validator


class MovieReviewData(BaseModel):
    review: str

    @validator('review')
    def validate_review(cls, string):
        if not isinstance(string, str):
            raise TypeError('`review` should be a strings')
        if not string or string.isspace() or string == 'nan':
            raise ValueError('`review` cannot be empty, contain only whitespace or NaN')
        return string


class MovieReviewResponse(BaseModel):
    probability: float
    sentiment: str
