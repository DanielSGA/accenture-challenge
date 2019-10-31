import React from 'react';
import { Image } from 'react-bootstrap';
export const Card = ({img})=>{
    return(
        <div className="card">
        <img src={img} alt="temp" className="imag"/>
          <div className="container">
            <h4><b>AC04U12</b></h4>
            <p>Color</p>
          </div>
        </div>
    )

}

export default Card;
